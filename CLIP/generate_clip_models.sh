git clone https://github.com/RobertBiehl/CLIP-tf2.git
conda deactivate #If needed
cd CLIP-tf2
pip install -r requirements.txt -U
python3 convert_clip.py --model "ViT-B/32" --output ../model
