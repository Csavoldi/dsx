from pathlib import Path
import subprocess,json,hashlib,datetime
root=Path('/home/lxe/gauntlet-comparison/20261025-low');root.mkdir(parents=True,exist_ok=False)
def tm(*args):return subprocess.check_output(['tmux',*args],text=True).strip()
# Restore the deleted Pi pane above the original preview, preserving the session.
preview='%226';local_pi=tm('split-window','-b','-v','-t',preview,'-p','58','-P','-F','#{pane_id}')
base=Path('/home/lxe/gauntlet-nanbeige-doom/20261025-exl3-thinking-off/TASK.single-line.txt').read_text().strip()
oldpath='/home/lxe/gauntlet-nanbeige-doom/20261025-exl3-thinking-off/game'
canonical=base.replace(oldpath,'{GAME}').replace('%226','{OUTPUT_PANE}')
for key,session,title,provider,url,model in [
 ('nanbeige-local','dsx-doom','Nanbeige EXL3 2.5 bpw | low','dsx-nanbeige','http://127.0.0.1:8095/v1','nanbeige4.3-33b'),
 ('nanbeige-cloud','nanbeige-doom-bf16','Nanbeige original 16-bit | low | Q8 KV','nanbeige-reference','http://127.0.0.1:8096/v1','Nanbeige4.3-33B'),
 ('deepseek-102','deepseek-doom-102','DeepSeek T16 | 2x3090 .102 | low','dsx-doom','http://192.168.0.102:8090/v1','deepseek-v4.1-flash')]:
 d=root/key
 for n in ['game','agent','sessions','receipts']:(d/n).mkdir(parents=True)
 if key=='nanbeige-local':pi,out=local_pi,preview
 else:
  pi=tm('new-session','-d','-s',session,'-x','54','-y','46','-P','-F','#{pane_id}','-c',str(d/'game'))
  out=tm('split-window','-v','-t',pi,'-p','42','-P','-F','#{pane_id}','-c',str(d/'game'))
 tm('set-window-option','-t',session+':0','window-size','manual');tm('resize-window','-t',session+':0','-x','54','-y','46')
 tm('set-window-option','-t',session+':0','pane-border-status','top')
 tm('set-option','-p','-t',pi,'pane-border-format',' Pi | '+title+' ')
 tm('set-option','-p','-t',out,'pane-border-format',' Doom preview | colored Braille | autoplay ')
 # Match the original preview's 17 rows exactly across all sessions.
 tm('resize-pane','-t',out,'-y','17')
 tm('select-pane','-t',pi)
 task=canonical.replace('{GAME}',str(d/'game')).replace('{OUTPUT_PANE}',out)
 (d/'TASK.single-line.txt').write_text(task+'\n');(d/'game/TASK.md').write_text(task+'\n')
 normalized=task.replace(str(d/'game'),'{GAME}').replace(out,'{OUTPUT_PANE}')
 assert normalized==canonical
 if key=='deepseek-102':
  config=json.loads(Path('/home/lxe/gauntlet-dsx-doom/20261025-termius-low-v3/agent/models.json').read_text())
 else:
  config=json.loads(Path('/home/lxe/nanbeige-runner/pi/agent/models.json').read_text())
  spec=config['providers'].pop('dsx-nanbeige');config['providers'][provider]=spec
 spec=config['providers'][provider];spec['baseUrl']=url;spec['models'][0]['id']=model;spec['models'][0]['name']=title
 spec['models'][0]['maxTokens']=65536;spec['models'][0]['contextWindow']=262144;spec['models'][0]['samplingParams']={'temperature':1.0,'top_p':0.95,'top_k':0,'min_p':0,'seed':1699757404}
 (d/'agent/models.json').write_text(json.dumps(config,indent=2)+'\n')
 settings=json.loads(Path('/home/lxe/nanbeige-runner/pi/agent/settings.json').read_text());settings['defaultThinkingLevel']='low';(d/'agent/settings.json').write_text(json.dumps(settings,indent=2)+'\n')
 script=f'''#!/bin/sh
set -eu
cd {d}/game
export PI_CODING_AGENT_DIR={d}/agent
unset NODE_OPTIONS
exec pi --offline --provider {provider} --model {model} --thinking low --tools read,bash,edit,write --append-system-prompt 'Use absolute paths for read, write, and edit tool calls. The game working directory is {d}/game.' --no-extensions --no-mcp --no-skills --no-prompt-templates --no-context-files --session-dir ../sessions "$(cat ../TASK.single-line.txt)"
'''
 (d/'run-pi.sh').write_text(script);(d/'run-pi.sh').chmod(0o755)
 manifest={'key':key,'session':session,'pi_pane':pi,'output_pane':out,'title':title,'provider':provider,'model':model,'base_url':url,'thinking_level':'low','reasoning_control':'native effort low' if key=='deepseek-102' else 'Nanbeige enable_thinking=true; no native low budget','context_capacity':262144,'max_output_tokens':65536,'temperature':1,'top_p':.95,'seed':1699757404,'normalized_prompt_sha256':hashlib.sha256(canonical.encode()).hexdigest(),'task_sha256':hashlib.sha256(task.encode()).hexdigest(),'terminal':'54x46','output_rows':tm('display-message','-p','-t',out,'#{pane_height}'),'fresh_game':True,'status':'prepared'}
 (d/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)
(root/'TASK.canonical.txt').write_text(canonical+'\n')
