download_unzip() {
    file_id="$1"

    gdown "$file_id" -O ./data/temp.zip

    unzip ./data/temp.zip -d ./data

    rm ./data/temp.zip
}

mkdir -p ./data

# keyframes_embs_clip_BigG2B.zip
# https://drive.google.com/file/d/1MsNT7cvu5K6EnV6L7fM36Vps822w7mMX/view?usp=drive_link
download_unzip "1MsNT7cvu5K6EnV6L7fM36Vps822w7mMX"

# keyframes_embs_clip_H.zip
# https://drive.google.com/file/d/1CLpotVpge9wbE51supRogxE5qPEoc-xy/view?usp=sharing
download_unzip "1CLpotVpge9wbE51supRogxE5qPEoc-xy"

# keyframes_embs_clip_S400M.zip
# https://drive.google.com/file/d/1zRyi_Ks-3Y3vJSXhGEIDtduq_vnkolND/view?usp=sharing
download_unzip "1zRyi_Ks-3Y3vJSXhGEIDtduq_vnkolND"
