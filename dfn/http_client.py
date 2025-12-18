"""
Client for dfn_2stage HTTP service
Demonstrates how to call the service and download results
"""

import requests
import argparse
import os
from pathlib import Path


def process_audio(service_url, audio_file, timeout=36000, export_stage1=False, output_dir='./audio_results'):
    """
    Send audio file to the HTTP service for processing
    
    Args:
        service_url: Base URL of the HTTP service (e.g., 'http://localhost:5000')
        audio_file: Path to the audio file
        timeout: Processing timeout in seconds
        export_stage1: Whether to export stage1 results
        output_dir: Directory to save output files
    
    Returns:
        dict with request_id and download URLs
    """
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        return None
    
    # Prepare the request
    files = {'audio': open(audio_file, 'rb')}
    data = {
        'timeout': str(timeout),
        'export_stage1': 'true' if export_stage1 else 'false'
    }
    
    try:
        print(f"Sending audio file to {service_url}/process...")
        print("NOTE: Large files will be processed in chunks to save VRAM. This may take some time.")
        print(f"Timeout set to {timeout} seconds.")
        
        response = requests.post(
            f"{service_url}/process",
            files=files,
            data=data,
            timeout=timeout + 30  # Add buffer to requests timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Processing successful!")
            print(f"Request ID: {result['request_id']}")
            print(f"Output files available at:")
            for name, url in result['files'].items():
                full_url = f"{service_url}{url}"
                print(f"  {name}: {full_url}")
            
            # Download files if output_dir is specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                download_files(service_url, result['request_id'], result['files'].keys(), output_dir)
            
            return result
        else:
            error = response.json()
            print(f"✗ Error ({response.status_code}): {error.get('error', 'Unknown error')}")
            return None
    
    except requests.exceptions.Timeout:
        print(f"✗ Request timeout after {timeout + 30}s")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def download_files(service_url, request_id, filenames, output_dir):
    """
    Download all output files from the service
    
    Args:
        service_url: Base URL of the HTTP service
        request_id: Request ID from processing
        filenames: List of filenames to download
        output_dir: Directory to save files to
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in filenames:
        try:
            url = f"{service_url}/download/{request_id}/{filename}.wav"
            print(f"Downloading {filename}...", end=' ')
            response = requests.get(url, timeout=60)
            
            if response.status_code == 200:
                output_path = os.path.join(output_dir, f"{filename}.wav")
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"✓ Saved to {output_path}")
            else:
                print(f"✗ Failed ({response.status_code})")
        except Exception as e:
            print(f"✗ Error: {e}")


def cleanup(service_url, request_id):
    """
    Clean up temporary files on the server
    
    Args:
        service_url: Base URL of the HTTP service
        request_id: Request ID from processing
    """
    try:
        response = requests.post(f"{service_url}/cleanup/{request_id}")
        if response.status_code == 200:
            print(f"✓ Cleaned up temporary files for request {request_id}")
        else:
            print(f"✗ Cleanup failed: {response.json()}")
    except Exception as e:
        print(f"✗ Error: {e}")


def health_check(service_url):
    """Check if the service is running"""
    try:
        response = requests.get(f"{service_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Service is healthy: {data['service']}")
            return True
        else:
            print(f"✗ Service returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot reach service: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Client for dfn_2stage HTTP service',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process audio file and download results
  python http_client.py --url http://localhost:5000 --audio input.wav --output ./results
  
  # Check service health
  python http_client.py --url http://localhost:5000 --health
  
  # Process with stage1 export and custom timeout
  python http_client.py --url http://localhost:5000 --audio input.wav --export-stage1 --timeout 180
        """
    )
    
    parser.add_argument('--url', default='http://localhost:5000',
                       help='Service URL (default: http://localhost:5000)')
    parser.add_argument('--audio', help='Audio file to process')
    parser.add_argument('--output', default='./audio_results',
                       help='Output directory for results (default: ./audio_results)')
    parser.add_argument('--timeout', type=int, default=36000,
                       help='Processing timeout in seconds (default: 36000)')
    parser.add_argument('--export-stage1', action='store_true',
                       help='Export stage1 non_speech result')
    parser.add_argument('--health', action='store_true',
                       help='Check service health and exit')
    parser.add_argument('--cleanup', help='Clean up temporary files for given request ID')
    
    args = parser.parse_args()
    
    # Check health
    if args.health:
        health_check(args.url)
        return
    
    # Cleanup
    if args.cleanup:
        cleanup(args.url, args.cleanup)
        return
    
    # Process audio
    if args.audio:
        result = process_audio(
            args.url,
            args.audio,
            timeout=args.timeout,
            export_stage1=args.export_stage1,
            output_dir=args.output
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
