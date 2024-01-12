cd ./../data                            && \
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1SD7NXIdWK97aFTKdePuw8qusMeGITrlu' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1SD7NXIdWK97aFTKdePuw8qusMeGITrlu" -O data.zip && rm -rf /tmp/cookies.txt  && \
unzip data.zip  && \
rm data.zip