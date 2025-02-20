#!/bin/bash

# then download subs, e.g.: https://www.opensubtitles.org/en/download/s/sublanguageid-eng/uploader-srjanapala/hearingimpaired-on/pimdbid-1091909/season-X
# put the zip in the subs folder and run this script

cd "$(dirname "$0")" || exit 1

mkdir -p subs 2>/dev/null
mkdir -p finished_translations 2>/dev/null

cd subs || exit 1
unzip *.zip || exit 1
mv */* . || exit 1
find . -type d -delete || exit 1
#rm *.HI.*.srt || exit 1

# Loop through all .srt files in the current directory
for file in *.srt; do
    # Extract the season and episode number using regex
    if [[ $file =~ S([0-9]+)E([0-9]+) ]]; then
        season=${BASH_REMATCH[1]}
        episode=${BASH_REMATCH[2]}
        
        # Pad both season and episode numbers with leading zeros if necessary
        padded_season=$(printf "%02d" "${season#0}")
        padded_episode=$(printf "%02d" "${episode#0}")
        
        # Construct the new filename
        new_name="S${padded_season}E${padded_episode}.srt"
        
        # Rename the file
        mv -v "$file" "$new_name"
    fi
done

mv *.zip ../finished_translations || exit 1
