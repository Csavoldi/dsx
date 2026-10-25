import {mkdtempSync,mkdirSync,writeFileSync,readFileSync,rmSync,copyFileSync} from 'node:fs';
import {tmpdir} from 'node:os';import {join} from 'node:path';import {createHash} from 'node:crypto';import assert from 'node:assert/strict';
const root=mkdtempSync(join(tmpdir(),'dsx-feedback-'));process.env.GAUNTLET_RUN=root;process.env.GAUNTLET_JUDGE_VERSION='8';
mkdirSync(join(root,'workspace'));writeFileSync(join(root,'workspace/render.py'),'# fixture\n');writeFileSync(join(root,'started.json'),JSON.stringify({pane:'%999'}));
const hash=createHash('sha256').update(readFileSync(join(root,'workspace/render.py'))).digest('hex');const handlers={};
copyFileSync(new URL('./acceptance-stop.ts',import.meta.url),join(root,'acceptance-stop.ts'));
const {default:install}=await import(new URL('file://'+join(root,'acceptance-stop.ts')));install({on:(name,f)=>handlers[name]=f,setActiveTools:()=>{}});
try{
 writeFileSync(join(root,'latest-check.json'),JSON.stringify({pass:false,judge_version:8,source_sha256:hash,renderer:{phases:[{max_dots:100,typical_width_fraction:.5,typical_height_fraction:.5,frame_certification:{supported_full_frame:true,shaded_frame_fraction:0,max_ghost_fraction:0}}]}}));
 const event={toolName:'bash',input:{command:'tmux capture-pane -p -t %999'},content:[{type:'text',text:'⣿'.repeat(30)}],isError:false,toolCallId:'capture'};
 const patched=handlers.tool_result(event);assert.equal(patched.content.length,2);assert.match(patched.content[1].text,/differentiated shading/);
 const toolMessage={role:'tool',content:patched.content.map(x=>x.text).join('\n')};
 const first={messages:[{role:'user',content:'task'},toolMessage]};assert.equal(handlers.before_provider_request({payload:first}),undefined);
 const next={messages:[...first.messages,{role:'assistant',content:'repair'},{role:'tool',content:'done'}]};assert.equal(handlers.before_provider_request({payload:next}),undefined);
 assert.deepEqual(next.messages.slice(0,first.messages.length),first.messages,'Feedback remains in the stored tool result prefix');
 assert.equal(handlers.tool_result(event),undefined,'No duplicate feedback for an unchanged source');
 writeFileSync(join(root,'workspace/render.py'),'# blank revision\n');
 const blankHash=createHash('sha256').update(readFileSync(join(root,'workspace/render.py'))).digest('hex');
 writeFileSync(join(root,'latest-check.json'),JSON.stringify({pass:false,judge_version:8,source_sha256:blankHash,renderer:{phases:[{max_dots:0,frame_certification:{supported_full_frame:true}}]}}));
 const blank=handlers.tool_result({...event,toolCallId:'blank',content:[{type:'text',text:' '.repeat(30)}]});assert.match(blank.content.at(-1).text,/No visible braille dots/);
 writeFileSync(join(root,'acceptance.json'),JSON.stringify({pass:true,judge_version:8,source_sha256:blankHash}));
 assert.equal(handlers.tool_call({}),undefined,'A blank self-check attempt cannot satisfy the successful-display guard');
 const journal=readFileSync(join(root,'functional-feedback.jsonl'),'utf8').trim().split('\n');assert.equal(journal.length,2);assert.equal(JSON.parse(journal[0]).delivery,'persisted-tool-result');
 console.log('Persistent feedback checks passed: tool-result delivery, unchanged follow-up prefix, one note per revision, no transient user-message injection.');
}finally{rmSync(root,{recursive:true,force:true});}
