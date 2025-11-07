from pathlib import Path
from enhancer_app.app import Job, _run_enhance

def main():
    f = Path('input_audio')/ 'Guest.WAV'
    if not f.exists():
        # try another present file
        files = list(Path('input_audio').glob('*.WAV')) + list(Path('input_audio').glob('*.wav'))
        if not files:
            raise SystemExit('No input files found in input_audio/*.wav')
        f = files[0]
    job = Job('testjob', [f])
    _run_enhance(job)
    print('JOB STATE:', job.state)
    if job.error:
        print('JOB ERROR:', job.error)
    print('OUTPUT DIR:', job.output_dir)

if __name__ == '__main__':
    main()

