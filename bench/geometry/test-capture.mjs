import {mkdtempSync,mkdirSync,writeFileSync,readFileSync,rmSync,existsSync} from 'node:fs';
import {tmpdir} from 'node:os';import {join} from 'node:path';import {createHash} from 'node:crypto';import assert from 'node:assert/strict';
const root=mkdtempSync(join(tmpdir(),'dsx-capture-'));process.env.GAUNTLET_RUN=root;process.env.GAUNTLET_JUDGE_VERSION='7';process.env.GAUNTLET_NORMALIZE_CAPTURE_BLANKS='1';
mkdirSync(join(root,'workspace'));writeFileSync(join(root,'workspace/render.py'),'# fixture\n');writeFileSync(join(root,'started.json'),JSON.stringify({pane:'%999'}));
const hash=createHash('sha256').update(readFileSync(join(root,'workspace/render.py'))).digest('hex');const handlers={};let tools=null;
const {default:install}=await import(new URL('./acceptance-stop.ts',import.meta.url));install({on:(name,f)=>handlers[name]=f,setActiveTools:v=>tools=v});
try{
 const raw='⠀'.repeat(30)+'⣿'.repeat(30)+'\n'+'⠿'.repeat(30)+'⠀'.repeat(30);
 const event={toolName:'bash',input:{command:'tmux capture-pane -p -t %999'},content:[{type:'text',text:raw}],isError:false,toolCallId:'capture'};
 assert.equal(handlers.tool_result({...event,input:{command:'cat render.py'}}),undefined,'Code and unrelated tools retain their original bytes');
 assert.equal(existsSync(join(root,'capture-normalization.jsonl')),false);
 writeFileSync(join(root,'acceptance.json'),JSON.stringify({pass:true,judge_version:5,source_sha256:hash}));
 const out=handlers.tool_result(event);assert.equal(tools,null,'Wrong judge version must not certify completion');
 assert.equal(out.content[0].text,raw.replaceAll('⠀',' '));
 assert.deepEqual(out.content[0].text.split('\n').map(x=>[...x].length),raw.split('\n').map(x=>[...x].length));
 assert.equal(event.content[0].text,raw,'Original tool event remains unchanged');
 const audit=JSON.parse(readFileSync(join(root,'capture-normalization.jsonl'),'utf8').trim());assert.equal(audit.blank_cells,60);assert.equal(audit.raw_content[0].text,raw);
 assert.equal(JSON.parse(readFileSync(join(root,'self-check.json'))).source_sha256,hash);
 writeFileSync(join(root,'acceptance.json'),JSON.stringify({pass:true,judge_version:7,source_sha256:hash}));
 assert.equal(handlers.tool_call({}).block,true);assert.deepEqual(tools,[]);
 console.log('Capture checks passed: named pane only, unchanged geometry/columns, raw audit preserved, matching judge required.');
}finally{rmSync(root,{recursive:true,force:true});}
