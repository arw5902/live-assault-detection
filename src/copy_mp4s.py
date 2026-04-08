#!/usr/bin/env python3

import os
import shutil
import argparse


def copy_mp4_files(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    copied = 0

    for root, dirs, files in os.walk(src_dir):
        for f in files:
            if f.lower().endswith(".mp4"):
                src_path = os.path.join(root, f)

                dst_path = os.path.join(dst_dir, f)

                # avoid overwriting duplicates
                if os.path.exists(dst_path):
                    name, ext = os.path.splitext(f)
                    i = 1
                    while True:
                        new_name = f"{name}_{i}{ext}"
                        dst_path = os.path.join(dst_dir, new_name)
                        if not os.path.exists(dst_path):
                            break
                        i += 1

                shutil.copy2(src_path, dst_path)
                copied += 1
                print(f"Copied: {src_path} -> {dst_path}")

    print(f"\nTotal files copied: {copied}")


def main():
    parser = argparse.ArgumentParser(description="Copy all MP4 files from a folder tree into another folder.")
    parser.add_argument("source", help="Source directory containing subfolders")
    parser.add_argument("destination", help="Destination directory for copied MP4 files")

    args = parser.parse_args()

    copy_mp4_files(args.source, args.destination)


if __name__ == "__main__":
    main()