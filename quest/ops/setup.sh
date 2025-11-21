mkdir -p build
cd build

# Rely on CMake to find the correct Python (set via Python_ROOT_DIR in CMakeLists.txt)
# and use its own logic to locate Torch. This avoids depending on the shell's `python`
# and works even if that Python does not have torch installed.
cmake ..
cmake --build . -j

echo "Compilation Finish"
cd ..
for file in $(find "./build" -maxdepth 1 -name \"*.so\"); do
    abs_file=$(realpath \"$file\")
    if [ -e \"$abs_file\" ]; then
        ln -s \"$abs_file\" ../
        echo \"Copied $abs_file...\"
    fi
done