// Sequential block uploads bound memory and retry only the interrupted block.
export async function uploadBlob(file, initialUrl, {renew, onProgress, xhrFactory = () => new XMLHttpRequest()}) {
  let uploadUrl = initialUrl;
  const blockSize = 4 * 1024 * 1024;
  const blocks = [];
  const nonce = crypto.randomUUID().replaceAll('-', '');
  const put = (url, body, headers, progress) => new Promise((resolve, reject) => {
    const xhr = xhrFactory();
    xhr.open('PUT', url);
    xhr.timeout = 120000;
    for (const [key, value] of Object.entries(headers)) xhr.setRequestHeader(key, value);
    xhr.upload.onprogress = event => {if (event.lengthComputable) progress(event.loaded);};
    xhr.onload = () => xhr.status >= 200 && xhr.status < 300 ? resolve() : reject({status:xhr.status});
    xhr.onerror = xhr.ontimeout = xhr.onabort = () => reject({status:0});
    xhr.send(body);
  });
  async function request(parameters, body, headers, progress = () => {}) {
    for (let attempt = 0; attempt < 4; attempt++) {
      const url = new URL(uploadUrl);
      for (const [key, value] of Object.entries(parameters)) url.searchParams.set(key, value);
      try {await put(url.href, body, headers, progress); return;}
      catch (error) {
        if (attempt === 3 || (error.status >= 400 && error.status < 500 && ![403,408,429].includes(error.status))) {
          throw new Error('uploadFailed');
        }
        if (error.status === 403) uploadUrl = (await renew()).upload_url;
        await new Promise(resolve => setTimeout(resolve, 500 * 2 ** attempt));
      }
    }
  }
  for (let offset = 0, index = 0; offset < file.size; offset += blockSize, index++) {
    const id = btoa(`${nonce}-${String(index).padStart(8, '0')}`);
    const block = file.slice(offset, offset + blockSize);
    await request({comp:'block', blockid:id}, block, {'Content-Type':'application/octet-stream'},
      bytes => onProgress(Math.min(file.size, offset + bytes), file.size));
    blocks.push(id);
    onProgress(Math.min(file.size, offset + block.size), file.size);
  }
  await request({comp:'blocklist'}, `<?xml version="1.0" encoding="utf-8"?><BlockList>${blocks.map(id => `<Latest>${id}</Latest>`).join('')}</BlockList>`,
    {'Content-Type':'application/xml', 'x-ms-blob-content-type':file.type || 'application/octet-stream'});
}
