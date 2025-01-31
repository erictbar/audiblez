#!/usr/bin/env python3
# audiblez - A program to convert e-books into audiobooks using
# Kokoro-82M model for high-quality text-to-speech synthesis.
# by Claudio Santini 2025 - https://claudio.uk
import argparse
import sys
import time
import shutil
import subprocess
import numpy as np
import soundfile
import ebooklib
import warnings
import re
import torch
from pathlib import Path
from string import Formatter
from bs4 import BeautifulSoup
from kokoro import KPipeline
from ebooklib import epub
from pydub import AudioSegment
from pick import pick
from tempfile import NamedTemporaryFile

from voices import voices, available_voices_str

sample_rate = 24000


def main(pipeline, file_path, voice, pick_manually, speed):
    filename = Path(file_path).name
    warnings.simplefilter("ignore")
    book = epub.read_epub(file_path)
    meta_title = book.get_metadata('DC', 'title')
    title = meta_title[0][0] if meta_title else ''
    meta_creator = book.get_metadata('DC', 'creator')
    by_creator = 'by ' + meta_creator[0][0] if meta_creator else ''

    cover_maybe = [c for c in book.get_items() if c.get_type() == ebooklib.ITEM_COVER]
    cover_image = cover_maybe[0].get_content() if cover_maybe else b""
    if cover_maybe:
        print(f'Found cover image {cover_maybe[0].file_name} in {cover_maybe[0].media_type} format')

    intro = f'{title} {by_creator}'
    print(intro)
    print('Found Chapters:', [c.get_name() for c in book.get_items() if c.get_type() == ebooklib.ITEM_DOCUMENT])
    if pick_manually:
        chapters = pick_chapters(book)
    else:
        chapters = find_chapters(book)
    print('Automatically selected chapters:', [c.get_name() for c in chapters])
    texts = extract_texts(chapters)

    has_ffmpeg = shutil.which('ffmpeg') is not None
    if not has_ffmpeg:
        print('\033[91m' + 'ffmpeg not found. Please install ffmpeg to create mp3 and m4b audiobook files.' + '\033[0m')

    total_chars, processed_chars = sum(map(len, texts)), 0
    print('Started at:', time.strftime('%H:%M:%S'))
    print(f'Total characters: {total_chars:,}')
    print('Total words:', len(' '.join(texts).split()))
    chars_per_sec = 500 if torch.cuda.is_available() else 50
    print(f'Estimated time remaining (assuming {chars_per_sec} chars/sec): {strfdelta((total_chars - processed_chars) / chars_per_sec)}')

    chapter_mp3_files = []
    for i, text in enumerate(texts, start=1):
        chapter_filename = filename.replace('.epub', f'_chapter_{i}.wav')
        chapter_mp3_files.append(chapter_filename)
        if Path(chapter_filename).exists():
            print(f'File for chapter {i} already exists. Skipping')
            continue
            
        if len(text.strip()) < 10:
            print(f'Skipping empty chapter {i}')
            chapter_mp3_files.remove(chapter_filename)
            continue
            
        print(f'Reading chapter {i} ({len(text):,} characters)...')
        if i == 1:
            text = intro + '.\n\n' + text
        start_time = time.time()

        audio_segments = gen_audio_segments(pipeline, text, voice, speed)
        if audio_segments:
            final_audio = np.concatenate(audio_segments)
            soundfile.write(chapter_filename, final_audio, sample_rate)
            end_time = time.time()
            delta_seconds = end_time - start_time
            chars_per_sec = len(text) / delta_seconds
            processed_chars += len(text)
            print(f'Estimated time remaining: {strfdelta((total_chars - processed_chars) / chars_per_sec)}')
            print('Chapter written to', chapter_filename)
            print(f'Chapter {i} read in {delta_seconds:.2f} seconds ({chars_per_sec:.0f} characters per second)')
            progress = processed_chars * 100 // total_chars
            print('Progress:', f'{progress}%\n')
        else:
            print(f'Warning: No audio generated for chapter {i}')
            chapter_mp3_files.remove(chapter_filename)

    if has_ffmpeg:
        create_index_file(title, by_creator, chapter_mp3_files)
        create_m4b(chapter_mp3_files, filename, title, by_creator, cover_image)


def gen_audio_segments(pipeline, text, voice, speed):
    audio_segments = []
    for gs, ps, audio in pipeline(text, voice=voice, speed=speed, split_pattern=r'\n+'):
        audio_segments.append(audio)
    return audio_segments


def extract_texts(chapters):
    texts = []
    for chapter in chapters:
        xml = chapter.get_body_content()
        soup = BeautifulSoup(xml, features='lxml')
        chapter_text = ''
        html_content_tags = ['title', 'p', 'h1', 'h2', 'h3', 'h4', 'li']
        for child in soup.find_all(html_content_tags):
            inner_text = child.text.strip() if child.text else ""
            if inner_text:
                chapter_text += inner_text + '\n'
        texts.append(chapter_text)
    return texts


def is_chapter(c):
    name = c.get_name().lower()
    return bool(
        'chapter' in name.lower()
        or re.search(r'part\d{1,3}', name)
        or re.search(r'ch\d{1,3}', name)
        or re.search(r'chap\d{1,3}', name)
    )


def find_chapters(book, verbose=False):
    chapters = [c for c in book.get_items() if c.get_type() == ebooklib.ITEM_DOCUMENT and is_chapter(c)]
    if verbose:
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                print(f"'{item.get_name()}'" + ', #' + str(len(item.get_body_content())))
    if len(chapters) == 0:
        print('Not easy to find the chapters, defaulting to all available documents.')
        chapters = [c for c in book.get_items() if c.get_type() == ebooklib.ITEM_DOCUMENT]
    return chapters


def pick_chapters(book):
    all_chapters_names = [c.get_name() for c in book.get_items() if c.get_type() == ebooklib.ITEM_DOCUMENT]
    title = 'Select which chapters to read in the audiobook'
    selected_chapters_names = pick(all_chapters_names, title, multiselect=True, min_selection_count=1)
    selected_chapters_names = [c[0] for c in selected_chapters_names]
    selected_chapters = [c for c in book.get_items() if c.get_name() in selected_chapters_names]
    return selected_chapters


def strfdelta(tdelta, fmt='{D:02}d {H:02}h {M:02}m {S:02}s'):
    remainder = int(tdelta)
    f = Formatter()
    desired_fields = [field_tuple[1] for field_tuple in f.parse(fmt)]
    possible_fields = ('W', 'D', 'H', 'M', 'S')
    constants = {'W': 604800, 'D': 86400, 'H': 3600, 'M': 60, 'S': 1}
    values = {}
    for field in possible_fields:
        if field in desired_fields and field in constants:
            values[field], remainder = divmod(remainder, constants[field])
    return f.format(fmt, **values)


def create_m4b(chapter_files, filename, title, author, cover_image):
    tmp_filename = filename.replace('.epub', '.tmp.opus')
    final_filename = filename.replace('.epub', '.m4b')

    if not Path(tmp_filename).exists():
        # Concat WAV files using ffmpeg
        with open('concat.txt', 'w') as f:
            for wav_file in chapter_files:
                f.write(f"file '{wav_file}'\n")
        
        print('Converting to Opus...')
        subprocess.run([
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', 'concat.txt',
            '-c:a', 'libopus',
            '-b:a', '64k',  # Lower bitrate for Opus
            '-application', 'audio',  # Optimize for music
            '-vbr', 'on',  # Variable bitrate
            '-compression_level', '10',  # Max compression
            tmp_filename
        ])
        Path('concat.txt').unlink()

    print('Creating M4B file...')
    cover_args = []
    if cover_image:
        cover_file = NamedTemporaryFile("wb", suffix='.png', delete=False)
        cover_file.write(cover_image)
        cover_file.close()
        cover_args = ['-i', cover_file.name]

    subprocess.run([
        'ffmpeg',
        '-i', tmp_filename,
        '-i', 'chapters.txt',
        *cover_args,
        '-map', '0:a',
        '-map_metadata', '1',
        '-metadata', f'title={title}',
        '-metadata', f'artist={author}',
        '-c:a', 'copy',
        '-disposition:v', 'attached_pic',
        '-f', 'mp4',
        final_filename
    ])

    # Cleanup
    Path(tmp_filename).unlink()
    if cover_image:
        Path(cover_file.name).unlink()

    print(f'{final_filename} created. Enjoy your audiobook.')
    print('Feel free to delete the intermediary .wav chapter files, the .m4b is all you need.')


def probe_duration(file_name):
    args = ['ffprobe', '-i', file_name, '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'default=noprint_wrappers=1:nokey=1']
    proc = subprocess.run(args, capture_output=True, text=True, check=True)
    return float(proc.stdout.strip())


def create_index_file(title, creator, chapter_mp3_files):
    with open("chapters.txt", "w") as f:
        f.write(f";FFMETADATA1\ntitle={title}\nartist={creator}\n\n")
        start = 0
        i = 0
        for c in chapter_mp3_files:
            duration = probe_duration(c)
            end = start + (int)(duration * 1000)
            f.write(f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={start}\nEND={end}\ntitle=Chapter {i}\n\n")
            i += 1
            start = end


def cli_main():
    voices_str = ', '.join(voices)
    epilog = ('example:\n' +
              '  audiblez book.epub -l en-us -v af_sky\n\n' +
              'available voices:\n' +
              available_voices_str)
    default_voice = 'af_sky'
    parser = argparse.ArgumentParser(epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('epub_file_path', help='Path to the epub file')
    parser.add_argument('-v', '--voice', default=default_voice, help=f'Choose narrating voice: {voices_str}')
    parser.add_argument('-p', '--pick', default=False, help=f'Interactively select which chapters to read in the audiobook', action='store_true')
    parser.add_argument('-s', '--speed', default=1.0, help=f'Set speed from 0.5 to 2.0', type=float)
    parser.add_argument('-c', '--cuda', default=False, help=f'Use GPU via Cuda in Torch if available', action='store_true')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    if args.cuda:
        if torch.cuda.is_available():
            print('CUDA GPU available')
            torch.set_default_device('cuda')
        else:
            print('CUDA GPU not available. Defaulting to CPU')

    pipeline = KPipeline(lang_code=args.voice[0])  # a for american or b for british
    main(pipeline, args.epub_file_path, args.voice, args.pick, args.speed)


if __name__ == '__main__':
    cli_main()
