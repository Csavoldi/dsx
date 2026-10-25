import {mkdtempSync,mkdirSync,writeFileSync,readFileSync,rmSync,existsSync} from 'node:fs';
import {tmpdir} from 'node:os';import {join} from 'node:path';import {createHash} from 'node:crypto';import assert from 'node:assert/strict';
const root=mkdtempSync(join(tmpdir(),'dsx-stop-'));process.env.GAUNTLET_RUN=root;mkdirSync(join(root,'workspace'));writeFileSync(join(root,'workspace/render.py'),'print("fixture")\n');writeFileSync(join(root,'started.json'),JSON.stringify({pane:'%999'}));
const hash=createHash('sha256').update(readFileSync(join(root,'workspace/render.py'))).digest('hex');const handlers={};let tools=null;
const {default:install}=await import(new URL('./acceptance-stop.ts',import.meta.url));install({on:(name,f)=>handlers[name]=f,setActiveTools:v=>tools=v});
try{
 assert.equal(handlers.tool_call({}),undefined);
 writeFileSync(join(root,'acceptance.json'),JSON.stringify({pass:true,judge_version:5,source_sha256:hash}));
 assert.equal(handlers.tool_call({}),undefined,'A mechanical pass without a self-check must not stop');
 handlers.tool_result({toolName:'bash',input:{command:'tmux capture-pane -p -t %999'},content:[],isError:true,toolCallId:'bad'});
 assert.equal(existsSync(join(root,'self-check.json')),false,'A failed display check must not count');
 writeFileSync(join(root,'acceptance.json'),JSON.stringify({pass:true,judge_version:5,source_sha256:'stale'}));
 handlers.tool_result({toolName:'bash',input:{command:'tmux capture-pane -p -t %999'},content:[{type:'text',text:'⣿'.repeat(30)}],isError:false,toolCallId:'good'});
 assert.equal(tools,null,'A stale certificate must not disable tools');
 writeFileSync(join(root,'acceptance.json'),JSON.stringify({pass:true,judge_version:5,source_sha256:hash}));
 handlers.before_provider_request({payload:{messages:[],tools:[]}});
 assert.deepEqual(tools,[]);assert.equal(handlers.tool_call({}).block,true);
 const out=handlers.before_provider_request({payload:{messages:[],tools:[{name:'write'}]}});assert.deepEqual(out.tools,[]);assert.equal(out.tool_choice,'none');assert.match(out.messages[0].content,/brief final summary/);
 assert.equal(JSON.parse(readFileSync(join(root,'accepted-stop.json'))).source_sha256,hash);
 console.log('Acceptance-stop checks passed: requires matching artifact, certification, and successful self-check; disables tools and requests final summary.');
}finally{rmSync(root,{recursive:true,force:true});}
