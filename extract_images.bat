cd /d "%~dp0data\real_dataset\images"
powershell -Command "Expand-Archive -Path archive.zip -DestinationPath ."
echo "Images extracted successfully!"
