@echo off
echo Creating conda environment 'EDU'...

conda create -n EDU python=3.9 -y

echo Activating environment...
call conda activate EDU

echo Installing PyTorch with CUDA support...
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

echo Installing additional dependencies...
pip install numpy pillow matplotlib tqdm scipy scikit-learn opencv-python

echo.
echo Environment setup complete!
echo.
echo To activate the environment, run:
echo conda activate EDU
echo.
echo Then you can run:
echo python train.py
echo python test.py
echo.
pause