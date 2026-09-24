"""YouTube input adapter. Download tools run in a cancellable subprocess."""
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlparse, parse_qs
from . import media
from .domain import ProviderError

VIDEO_ID = re.compile(r'^[A-Za-z0-9_-]{11}$')


def youtube_url(value):
    try:
        url = urlparse(value.strip())
        if url.scheme not in ('http','https') or url.username or url.password or url.port not in (None,80,443):
            raise ValueError()
        host = (url.hostname or '').lower()
        parts = url.path.strip('/').split('/')
        if host in ('youtu.be','www.youtu.be') and len(parts)==1:
            video_id = parts[0]
        elif host in ('youtube.com','www.youtube.com','m.youtube.com','music.youtube.com'):
            if url.path == '/watch':
                values = parse_qs(url.query).get('v',[])
                if len(values) != 1:
                    raise ValueError()
                video_id = values[0]
            elif len(parts)==2 and parts[0] in ('shorts','embed','live'):
                video_id = parts[1]
            else:
                raise ValueError()
        else:
            raise ValueError()
        if not VIDEO_ID.fullmatch(video_id):
            raise ValueError()
    except (ValueError,AttributeError):
        raise ValueError('请输入有效的单个 YouTube 视频链接（支持 watch、youtu.be 和 Shorts）')
    # Never fetch the caller's host, path, playlist, redirect or query parameters.
    return 'https://www.youtube.com/watch?v='+video_id


def validate_info(info, max_seconds):
    if not isinstance(info,dict):
        raise ProviderError('YouTube 返回了无法解析的视频信息')
    if info.get('_type','video') != 'video' or info.get('entries') is not None:
        raise ProviderError('仅支持单个视频，不支持频道或播放列表')
    if info.get('is_live') or info.get('live_status') in ('is_live','is_upcoming','post_live'):
        raise ProviderError('暂不支持直播中、尚未开始或尚未处理完成的视频')
    duration = info.get('duration')
    if not isinstance(duration,(int,float)) or not math.isfinite(duration) or not 0 < duration <= max_seconds:
        raise ProviderError('视频时长未知或超过配置的时长限制')
    if info.get('availability') in ('private','premium_only','subscriber_only','needs_auth'):
        raise ProviderError('该视频需要登录或会员权限，请上传本地视频文件')


def validate_video(path, config):
    if not path.is_file() or not 0 < path.stat().st_size <= config.max_upload_mb*1024**2:
        raise ProviderError('下载视频为空或超过配置的大小限制')
    info = media.probe(path)
    duration = float(info['format']['duration'])
    if not math.isfinite(duration) or not 0 < duration <= config.max_video_seconds:
        raise ProviderError('下载视频超过配置的时长限制')
    if not {'video','audio'} <= {s.get('codec_type') for s in info['streams']}:
        raise ProviderError('下载结果必须同时包含画面和音轨')
    return duration


def download_error(text):
    lowered = text.lower()
    if any(term in lowered for term in ('sign in','login','private video','members-only','age-restricted','confirm your age')):
        return 'YouTube 要求登录、年龄验证或访问权限，请改用本地文件上传'
    if any(term in lowered for term in ('video unavailable','not available','removed','copyright')):
        return '该 YouTube 视频不可用、已移除或在当前地区不可访问'
    if 'max-filesize' in lowered or 'larger than' in lowered:
        return 'YouTube 视频超过配置的大小限制'
    return 'YouTube 下载失败，请检查链接和网络；也可上传本地文件，或更新 yt-dlp 后重试'


class YouTubeSource:
    def __init__(self, config):
        self.config = config

    def _run(self, arguments, work, check_cancel, deadline):
        stdout, stderr = work/'stdout.log', work/'stderr.log'
        base = [sys.executable,'-m','yt_dlp','--ignore-config','--no-playlist',
                '--no-plugin-dirs','--no-cache-dir','--no-progress','--no-colors',
                '--js-runtimes','node','--no-remote-components','--socket-timeout','20',
                '--retries','2','--fragment-retries','2','--use-extractors','Youtube']
        proc = None
        with stdout.open('wb') as out, stderr.open('wb') as err:
            try:
                proc = subprocess.Popen(base+arguments,stdout=out,stderr=err,cwd=work,
                    start_new_session=True,env={'PATH':os.environ.get('PATH',''),
                    'HOME':str(work),'LANG':'C.UTF-8','PYTHONUNBUFFERED':'1'})
                while proc.poll() is None:
                    check_cancel()
                    if time.monotonic() > deadline:
                        raise ProviderError('YouTube 下载超时，请重试或上传本地文件')
                    if stdout.stat().st_size > 16*1024**2 or stderr.stat().st_size > 2*1024**2:
                        raise ProviderError('YouTube 返回了过大的响应')
                    # Download + merged output may coexist temporarily.
                    size = 0
                    for path in work.iterdir():
                        try:
                            size += path.stat().st_size
                        except FileNotFoundError:
                            pass  # yt-dlp may rename/delete a fragment while we inspect it.
                    if size > 3*self.config.max_upload_mb*1024**2+20*1024**2:
                        raise ProviderError('YouTube 下载临时文件超过大小限制')
                    time.sleep(.25)
                check_cancel()
                if proc.returncode:
                    raise ProviderError(download_error(stderr.read_text(encoding='utf-8',errors='replace')))
                return stdout.read_text(encoding='utf-8')
            finally:
                if proc is not None and proc.poll() is None:
                    # Kill the group, including an active FFmpeg merger.
                    try:
                        os.killpg(proc.pid,signal.SIGTERM)
                        proc.wait(timeout=3)
                    except (ProcessLookupError,subprocess.TimeoutExpired):
                        if proc.poll() is None:
                            os.killpg(proc.pid,signal.SIGKILL)
                            proc.wait()

    def download(self, url, destination, check_cancel, report):
        canonical = youtube_url(url)
        destination.parent.mkdir(parents=True,exist_ok=True)
        deadline = time.monotonic()+self.config.youtube_download_timeout_seconds
        with tempfile.TemporaryDirectory(prefix='.youtube-',dir=destination.parent) as directory:
            work = Path(directory)
            report('import',1)
            raw = self._run(['--skip-download','--dump-single-json',canonical],work,check_cancel,deadline)
            try:
                info = json.loads(raw)
            except (ValueError,TypeError) as exc:
                raise ProviderError('YouTube 返回了无法解析的视频信息') from exc
            validate_info(info,self.config.max_video_seconds)
            if info.get('id') != canonical.rsplit('=',1)[1]:
                raise ProviderError('YouTube 返回的视频与请求不一致')
            metadata = work/'video-info.json'
            metadata.write_text(raw,encoding='utf-8')
            report('import',2)
            self._run(['--load-info-json',str(metadata),'--no-simulate',
                '--format',f'bv*[height<={self.config.youtube_max_height}][ext=mp4]+ba[ext=m4a]/b[height<={self.config.youtube_max_height}][ext=mp4]/bv*[height<={self.config.youtube_max_height}]+ba/b[height<={self.config.youtube_max_height}]',
                '--merge-output-format','mp4','--remux-video','mp4',
                '--max-filesize',str(self.config.max_upload_mb*1024**2),
                '--output',str(work/'source.%(ext)s')],work,check_cancel,deadline)
            check_cancel()
            downloaded = work/'source.mp4'
            duration = validate_video(downloaded,self.config)
            downloaded.replace(destination)
            report('import',4)
            title = re.sub(r'[\x00-\x1f/\\]',' ',str(info.get('title') or info['id'])).strip()[:180]
            return {'kind':'youtube','url':canonical,'video_id':info['id'],'title':title,
                    'duration':duration,'downloaded':True}


def prepare_input(job, config, storage, check_cancel, report, adapter=None):
    key = f'jobs/{job.id}/source.json'
    if not storage.exists(key):
        return None  # Local upload already prepared its input.
    source = storage.read_json(key)
    if source.get('kind') != 'youtube':
        raise ProviderError('不支持此视频来源')
    destination = storage.path(job.input_key)
    if destination.is_file():
        # A crash after atomic move must not trigger another download.
        validate_video(destination,config)
        source['downloaded'] = True
    else:
        source = (adapter or YouTubeSource(config)).download(source['url'],destination,check_cancel,report)
    storage.write_json(key,source)
    return source
