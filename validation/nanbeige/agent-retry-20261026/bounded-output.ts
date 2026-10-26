import fs from 'node:fs';
import path from 'node:path';
const root='/home/lxe/gauntlet-nanbeige-doom/20261026-config-retry/receipts/tool-output';
export default function(pi: any) {
  pi.on('tool_result', (event: any) => {
    if (event.toolName!=='bash' || !Array.isArray(event.content)) return;
    const raw=event.content.filter((c:any)=>c.type==='text').map((c:any)=>c.text).join('\n');
    const plain=raw.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g,'').replace(/[\x00-\x08\x0b-\x1f\x7f]/g,'');
    const animation=(raw.match(/\x1b\[H/g)||[]).length>2 || (plain.match(/[\u2800-\u28ff]/g)||[]).length>200;
    if (!animation && plain.length<=4096) return;
    fs.mkdirSync(root,{recursive:true});
    const file=path.join(root,`${String(event.toolCallId||Date.now()).replace(/[^\w-]/g,'_')}.txt`);
    fs.writeFileSync(file,raw);
    const body=animation ? plain.split('\n').slice(-6).map(x=>x.slice(0,120)).join('\n') : plain.slice(0,1800)+'\n...\n'+plain.slice(-1800);
    return {content:[{type:'text',text:`[Output bounded: ${raw.length} characters; full tool output saved at ${file}. ${animation?'Terminal animation omitted; inspect the tmux preview instead.':''}]\n${body}`} ]};
  });
}
