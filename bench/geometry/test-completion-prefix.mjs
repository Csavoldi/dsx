import {mkdtempSync,mkdirSync,writeFileSync,readFileSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';import {join} from 'node:path';import {createHash} from 'node:crypto';import assert from 'node:assert/strict';
const root=mkdtempSync(join(tmpdir(),'dsx-completion-'));process.env.GAUNTLET_RUN=root;process.env.GAUNTLET_JUDGE_VERSION='8';process.env.GAUNTLET_PRESERVE_COMPLETION_PREFIX='1';
mkdirSync(join(root,'workspace'));writeFileSync(join(root,'workspace/render.py'),'# fixture\n');const hash=createHash('sha256').update(readFileSync(join(root,'workspace/render.py'))).digest('hex');const handlers={};let active;
const {default:install}=await import(new URL('./acceptance-stop.ts',import.meta.url));install({on:(name,f)=>handlers[name]=f,setActiveTools:v=>active=v});
try{
 const definitions=[{type:'function',function:{name:'bash',parameters:{type:'object'}}}];const saved=structuredClone(definitions);
 assert.equal(handlers.before_provider_request({payload:{messages:[],tools:definitions,reasoning_effort:'low'}}),undefined);
 definitions[0].function.name='mutated';
 writeFileSync(join(root,'acceptance.json'),JSON.stringify({pass:true,judge_version:8,source_sha256:hash}));writeFileSync(join(root,'self-check.json'),JSON.stringify({time:1,source_sha256:hash}));
 const output=handlers.before_provider_request({payload:{messages:[{role:'tool',content:'checked'}],tools:[],reasoning_effort:'low'}});
 assert.deepEqual(active,[],'Execution tools remain disabled');assert.deepEqual(output.tools,saved,'Native tool header is preserved from the prior payload');assert.equal(output.tool_choice,'auto');assert.equal(output.reasoning_effort,'low','Thinking header must stay unchanged');assert.match(output.messages.at(-1).content,/do not.*call further tools/);assert.equal(handlers.tool_call({}).block,true,'Further tool execution remains blocked');
 console.log('Completion-prefix guard: preserves frozen tool definitions and reasoning header while blocking execution.');
}finally{rmSync(root,{recursive:true,force:true});}
