const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const {chromium} = require('playwright');

const root = path.resolve(__dirname, '..');
const html = `<!doctype html><html><head><style>
*{box-sizing:border-box}body{margin:0;background:#333;font:14px sans-serif;color:white}
button,input,select{font:inherit}button{cursor:pointer;padding:5px;border-radius:4px}
button:disabled{opacity:.45;cursor:default}.dialog{position:fixed;left:50%;top:50%;transform:translate(-50%,-50%);width:460px;max-width:calc(100vw - 16px);padding:16px;background:#252525;border:1px solid #777;border-radius:6px;z-index:11000}
table{width:100%;margin:12px 0}td{padding:5px}label{display:block;margin-top:8px}select,input[type=text]{width:100%;margin:5px 0;padding:6px}.dialog footer{display:flex;gap:8px;margin-top:12px}
#map{position:fixed;inset:0;background-image:url('/maps/dnd1.jpg');background-size:cover;opacity:.7}
</style></head><body><div id="map"></div><script>
Math.clamp=(value,min,max)=>Math.min(max,Math.max(min,value));
window.CONFIG={Canvas:{minZoom:0.1}};
const hooks=new Map();window.Hooks={once(name,callback){hooks.set(name,callback)},on(name,callback){hooks.set(name,callback)}};
const settings=new Map();window.game={settings:{register(_m,key,value){settings.set(key,value.default)},get(_m,key){return settings.get(key)},async set(_m,key,value){settings.set(key,value)}},modules:new Map([['sarween',{version:'1.3.0'}]])};
game.user={isGM:true};
window.ui={notifications:{info(){},warn(message){window.lastWarning=message},error(message){window.lastWarning=message}}};
const names=['Red','Blue','Yellow','Green','White'];
const tokens=names.map((name,index)=>({id:String(index),name,x:index*50,y:0,width:1,height:1,actor:{system:{attributes:{movement:{walk:30}}}},sight:{enabled:true,range:30},async update(values){Object.assign(this,values);window.lastTokenMove={name,...values}}}));tokens.get=id=>tokens.find(token=>token.id===id);
const scale=Math.min(innerWidth/2400,innerHeight/1350);
window.canvas={ready:true,scene:{id:'scene',width:2400,height:1350,grid:{size:50,distance:5,type:1},tokenVision:true,fog:{exploration:true},tokens},grid:{measurePath(points){return {distance:Math.max(Math.abs(points[1].x-points[0].x),Math.abs(points[1].y-points[0].y))/50*5}}},dimensions:{sceneX:0,sceneY:0},stage:{worldTransform:{a:scale,b:0,c:0,d:scale,tx:0,ty:0},pivot:{x:0,y:0}},tokens:{releaseAll(){},get(id){return {document:tokens.get(id),control(){window.controlled=(window.controlled||new Set());window.controlled.add(id)}}}},pan(){},async animatePan(){}};
window.Dialog=class{constructor(config){this.config=config}render(){const element=document.createElement('section');element.className='dialog';element.innerHTML='<strong>'+this.config.title+'</strong>'+this.config.content+'<footer></footer>';for(const [key,button]of Object.entries(this.config.buttons)){const control=document.createElement('button');control.textContent=button.label;control.dataset.button=key;control.onclick=()=>{button.callback?.([element]);element.remove()};element.querySelector('footer').append(control)}document.body.append(element);return this}};
window.WebSocket=class{static OPEN=1;static CONNECTING=0;constructor(){this.readyState=1;setTimeout(()=>this.onopen?.(),0)}send(raw){const message=JSON.parse(raw);window.messages=(window.messages||[]);window.messages.push(message);if(message.type==='captureControl'){const state={start:'recording',confirm:'confirmed',stop:'saved',resume:'idle'}[message.action];if(state)setTimeout(()=>this.onmessage?.({data:JSON.stringify({type:'captureStatus',state,sessionId:message.sessionId,index:message.index,reason:message.reason})}),0)}}close(){this.readyState=3}};
</script><script type="module">import '/module.js';hooks.get('init')();hooks.get('ready')();window.mockReady=true;</script></body></html>`;

(async () => {
  new (require('node:vm').Script)(html.match(/<script>([\s\S]*?)<\/script>/)[1]);
  let browser;
  let deadline;
  const server = http.createServer((request, response) => {
    if (request.url === '/') {response.setHeader('Content-Type', 'text/html');response.end(html);return;}
    const relative = decodeURIComponent(request.url).replace(/^\/modules\/sarween\//, '').replace(/^\//, '');
    const allowed = ['module.js', 'capture_logic.mjs', 'movement_logic.mjs', 'maps/dnd1.jpg', ...[10,11,12,13].map(id=>`viewport_markers/marker_${id}.png`)];
    if (!allowed.includes(relative)) {response.statusCode=404;response.end();return;}
    response.setHeader('Content-Type', relative.endsWith('.js')||relative.endsWith('.mjs')?'text/javascript':relative.endsWith('.png')?'image/png':'image/jpeg');
    response.end(fs.readFileSync(path.join(root,relative)));
  });
  try {
    await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
    browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
    deadline=setTimeout(()=>browser.close(),45000);
    process.once('SIGINT',()=>browser.close());
    for (const viewport of [{width:1920,height:1080},{width:390,height:844}]) {
      const page=await browser.newPage({viewport});
      page.setDefaultTimeout(5000);
      const errors=[];page.on('pageerror',error=>{errors.push(error.message);console.error('Browser error:',error.message)});
      await page.goto('http://127.0.0.1:'+server.address().port);
      await page.waitForFunction(()=>window.mockReady,{},{timeout:5000});
      await page.evaluate(()=>hooks.get('controlToken')({document:tokens[0]},true));
      await page.getByText('Red: 0/30 ft',{exact:true}).waitFor();
      await page.evaluate(()=>{
        for(let column=1;column<=7;column+=1)hooks.get('updateToken')(tokens[0],{x:column*50,y:0});
      });
      await page.getByText('Red: 5 ft over',{exact:true}).waitFor();
      assert((await page.locator('#sarween-movement-overlay').innerHTML()).includes('#ef4444'));
      await page.evaluate(()=>hooks.get('controlToken')({document:tokens[0]},false));
      await page.locator('#sarween-movement-overlay').waitFor({state:'detached'});
      await page.locator('#sarween-two-mini-test-btn').click();
      assert.equal(await page.locator('.dialog input[type=checkbox]:checked').count(),5);
      await page.locator('[data-button=start]').click();
      await page.locator('[data-action=placed]:not([disabled])').waitFor({timeout:8000});
      assert.equal(await page.locator('#sarween-test-targets > div').count(),1);
      assert.equal(await page.evaluate(()=>window.controlled.size),5);
      assert.equal(await page.evaluate(()=>window.messages.find(message=>message.action==='start').targets.length),30);
      for (const selector of ['#sarween-capture','#sarween-status']) {
        const box=await page.locator(selector).boundingBox();
        assert(box.x>=0 && box.y>=0 && box.x+box.width<=viewport.width+1 && box.y+box.height<=viewport.height+1,`${selector} is out of bounds`);
      }
      const tool=await page.locator('#sarween-capture').boundingBox();
      const bar=await page.locator('#sarween-status').boundingBox();
      assert(tool.y+tool.height<=bar.y, 'Capture tool overlaps status bar');
      for (const marker of await page.locator('#sarween-viewport-markers img').all()) {
        const box=await marker.boundingBox();
        assert(tool.x+tool.width<=box.x || tool.x>=box.x+box.width || tool.y+tool.height<=box.y || tool.y>=box.y+box.height,'Capture tool overlaps an ArUco marker');
      }
      await page.screenshot({path:`/tmp/sarween-capture-${viewport.width}.png`});
      await page.locator('[data-action=placed]').click();
      await page.waitForFunction(()=>window.lastTokenMove?.name==='Red');
      await page.locator('[data-action=stop]').click();
      await page.getByText('Partial recording saved',{exact:true}).waitFor();
      await page.locator('[data-action=resume]').click();
      await page.locator('#sarween-capture').waitFor({state:'detached'});
      assert.deepEqual(errors,[]);
      await page.close();
    }
    console.log('PASS: capture setup, token vision selection, confirmation, stop/resume, desktop/mobile bounds');
  } finally {
    clearTimeout(deadline);
    if (browser) await browser.close();
    await new Promise(resolve=>server.close(resolve));
  }
})().catch(error=>{console.error(error);process.exitCode=1});
