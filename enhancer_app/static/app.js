const drop = document.getElementById('drop');
const fileIn = document.getElementById('file');
const log = document.getElementById('log');

function println(msg){
  log.textContent += msg + "\n";
}

function prevent(e){ e.preventDefault(); e.stopPropagation(); }

['dragenter','dragover'].forEach(ev=>drop.addEventListener(ev, e=>{prevent(e); drop.classList.add('hover');}));
['dragleave','drop'].forEach(ev=>drop.addEventListener(ev, e=>{prevent(e); drop.classList.remove('hover');}));

drop.addEventListener('click', ()=> fileIn.click());
fileIn.addEventListener('change', ()=>{
  if(fileIn.files?.length) handleFiles(fileIn.files);
});

drop.addEventListener('drop', async (e)=>{
  const files = [...(e.dataTransfer?.files||[])];
  if(files.length===0) return;
  handleFiles(files);
});

async function handleFiles(files){
  println(`Uploading ${files.length} file(s)...`);
  const fd = new FormData();
  for(const f of files){ fd.append('files', f, f.name); }
  const res = await fetch('/api/upload', {method:'POST', body:fd});
  if(!res.ok){ println('Upload failed.'); return; }
  const {job_id} = await res.json();
  println(`Job ${job_id} queued.`);
  poll(job_id);
}

// Create a simple progress bar element
function ensureBar(){
  let bar = document.getElementById('bar');
  if(!bar){
    bar = document.createElement('div');
    bar.id = 'bar';
    bar.style.cssText = 'height:10px;background:#223;border:1px solid #345;border-radius:6px;overflow:hidden;margin-top:8px;';
    const fill = document.createElement('div');
    fill.id = 'barfill';
    fill.style.cssText = 'height:100%;width:0%;background:#4aa;transition:width .25s linear;';
    bar.appendChild(fill);
    log.appendChild(bar);
  }
}

function setProgress(completed, total){
  ensureBar();
  const pct = total>0? Math.floor((completed/total)*100) : 0;
  const fill = document.getElementById('barfill');
  if(fill){ fill.style.width = pct + '%'; }
}

async function poll(job){
  while(true){
    await new Promise(r=>setTimeout(r, 800));
    const res = await fetch(`/api/status/${job}`);
    if(!res.ok){ println('Status error.'); return; }
    const s = await res.json();
    if(s.state==='queued') println('Waiting...');
    if(s.state==='running'){
      const msg = `Enhancing ${s.completed ?? 0}/${s.total ?? '?'}...`;
      println(msg);
      setProgress(s.completed||0, s.total||0);
    }
    if(s.state==='done'){
      setProgress(s.total||1, s.total||1);
      println(`Done. Output: ${s.output_dir}`);
      const btn = document.createElement('button');
      btn.textContent = 'Open Output Folder';
      btn.onclick = async ()=>{ await fetch(`/api/open/${job}`, {method:'POST'}); };
      log.appendChild(btn);
      return;
    }
    if(s.state==='failed'){
      println(`Failed: ${s.error||'unknown error'}`);
      return;
    }
  }
}
