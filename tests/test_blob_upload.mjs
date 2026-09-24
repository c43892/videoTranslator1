import assert from 'node:assert/strict';
import {uploadBlob} from '../packages/videotranslator/videotranslator/web/blob-upload.js';

const requests = [];
let renewals = 0;
class Request {
  upload = {};
  headers = {};
  open(method, url) {this.method = method; this.url = new URL(url);}
  setRequestHeader(key, value) {this.headers[key] = value;}
  send(body) {
    this.body = body;
    requests.push(this);
    this.status = requests.length === 1 ? 403 : 201;
    this.onload();
  }
}
const file = new Blob([new Uint8Array(5 * 1024 * 1024)], {type:'video/mp4'});
await uploadBlob(file, 'https://test.blob.core.windows.net/uploads/input?sig=old', {
  renew: async () => {renewals++; return {upload_url:'https://test.blob.core.windows.net/uploads/input?sig=new'};},
  onProgress: () => {}, xhrFactory: () => new Request()
});
assert.equal(renewals, 1);
assert.equal(requests.length, 4); // Failed block, retry, final block, commit.
assert.equal(requests[0].url.searchParams.get('blockid'), requests[1].url.searchParams.get('blockid'));
assert.equal(requests[1].url.searchParams.get('sig'), 'new');
assert.equal(requests[1].body.size, 4 * 1024 * 1024);
assert.equal(requests[2].body.size, 1024 * 1024);
assert.equal(requests[3].url.searchParams.get('comp'), 'blocklist');
assert.equal(requests[3].headers['x-ms-blob-content-type'], 'video/mp4');
assert.ok(requests[3].body.includes(requests[2].url.searchParams.get('blockid')));
assert.ok(requests.every(request => !request.headers.Authorization));
console.log('Blob block upload retry and SAS renewal passed');
