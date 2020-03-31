wget -O ./UTD/RGB.zip http://www.utdallas.edu/~kehtar/UTD-MAD/RGB.zip
chmod +x ./UTD/RGB.zip
unzip ./UTD/RGB.zip -d ./UTD/
rm ./UTD/RGB.zip
python ./UTD/process_UTD.py
