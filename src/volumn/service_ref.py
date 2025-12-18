"""
HTTP service wrapper for dfn_2stage.py
Provides REST API endpoints to call the audio separation functionality over HTTP
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from flask import Flask, request, jsonify, send_file
import argparse

# Import the main function from dfn_2stage
from dfn_2stage import main as dfn_main

app = Flask(__name__)

# Configuration
UPLOAD_DIR = tempfile.gettempdir()
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB


def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'wav', 'mp3', 'flac', 'ogg'}


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'dfn_2stage HTTP service'
    })


@app.route('/process', methods=['POST'])
def process_audio():
    """
    Process audio file and separate speech from non-speech
    
    Expected request:
    - multipart/form-data with audio file
    - Optional parameters:
      - timeout: timeout in seconds (default: 120)
      - export_stage1: export stage1 result (default: false)
    
    Returns:
    - JSON with paths to output files
    """
    try:
        # Check if audio file is present
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(audio_file.filename):
            return jsonify({'error': 'File type not allowed. Use: wav, mp3, flac, ogg'}), 400
        
        # Check file size
        audio_file.seek(0, os.SEEK_END)
        file_size = audio_file.tell()
        audio_file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({'error': f'File too large. Max: {MAX_FILE_SIZE / 1024 / 1024:.0f} MB'}), 400
        
        # Get parameters
        timeout = float(request.form.get('timeout', 120))
        export_stage1 = request.form.get('export_stage1', 'false').lower() == 'true'
        
        # Create temporary directories for this request
        request_id = os.urandom(8).hex()
        request_dir = os.path.join(UPLOAD_DIR, f'dfn_{request_id}')
        os.makedirs(request_dir, exist_ok=True)
        
        try:
            # Save uploaded file
            input_path = os.path.join(request_dir, 'input_audio.wav')
            audio_file.save(input_path)
            
            # Create output directory
            output_dir = os.path.join(request_dir, 'output')
            os.makedirs(output_dir, exist_ok=True)
            
            # Run dfn_2stage
            import sys
            original_argv = sys.argv
            sys.argv = [
                'dfn_2stage.py',
                '--in', input_path,
                '--outdir', output_dir,
                '--timeout', str(timeout),
            ]
            if export_stage1:
                sys.argv.append('--export_stage1')
            
            try:
                dfn_main()
            finally:
                sys.argv = original_argv
            
            # Prepare output files
            output_files = {
                'speech': os.path.join(output_dir, 'speech.wav'),
                'non_speech': os.path.join(output_dir, 'non_speech.wav'),
            }
            
            if export_stage1:
                output_files['non_speech_stage1'] = os.path.join(output_dir, 'non_speech_stage1.wav')
            
            # Check if output files were created
            missing_files = [k for k, v in output_files.items() if not os.path.exists(v)]
            if missing_files:
                return jsonify({
                    'error': f'Processing failed. Missing files: {", ".join(missing_files)}'
                }), 500
            
            # Return file paths and download URLs
            result = {
                'status': 'success',
                'request_id': request_id,
                'files': {
                    'speech': f'/download/{request_id}/speech.wav',
                    'non_speech': f'/download/{request_id}/non_speech.wav',
                }
            }
            
            if export_stage1:
                result['files']['non_speech_stage1'] = f'/download/{request_id}/non_speech_stage1.wav'
            
            # Store the request_dir in app config for later download
            if not hasattr(app, 'temp_dirs'):
                app.temp_dirs = {}
            app.temp_dirs[request_id] = request_dir
            
            return jsonify(result), 200
        
        except Exception as e:
            return jsonify({
                'error': str(e),
                'error_type': type(e).__name__
            }), 500
    
    except Exception as e:
        return jsonify({
            'error': f'Request processing failed: {str(e)}',
            'error_type': type(e).__name__
        }), 500


@app.route('/download/<request_id>/<filename>', methods=['GET'])
def download_file(request_id, filename):
    """
    Download processed audio file
    """
    try:
        if not hasattr(app, 'temp_dirs') or request_id not in app.temp_dirs:
            return jsonify({'error': 'Request not found or already cleaned up'}), 404
        
        request_dir = app.temp_dirs[request_id]
        output_dir = os.path.join(request_dir, 'output')
        
        # Validate filename to prevent directory traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({'error': 'Invalid filename'}), 400
        
        file_path = os.path.join(output_dir, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': f'File not found: {filename}'}), 404
        
        return send_file(file_path, as_attachment=True, download_name=filename)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/cleanup/<request_id>', methods=['POST'])
def cleanup(request_id):
    """
    Clean up temporary files for a request
    """
    try:
        if not hasattr(app, 'temp_dirs') or request_id not in app.temp_dirs:
            return jsonify({'error': 'Request not found'}), 404
        
        request_dir = app.temp_dirs[request_id]
        shutil.rmtree(request_dir, ignore_errors=True)
        del app.temp_dirs[request_id]
        
        return jsonify({'status': 'cleaned up'}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500


def main():
    parser = argparse.ArgumentParser(description='HTTP service for dfn_2stage audio processing')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    print(f"Starting dfn_2stage HTTP service on {args.host}:{args.port}")
    print(f"API endpoint: http://{args.host}:{args.port}/process")
    print(f"Health check: http://{args.host}:{args.port}/health")
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
