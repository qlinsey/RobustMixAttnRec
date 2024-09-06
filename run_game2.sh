echo lmd:0.1 sigma: 0.02 norm: false
python run.py --templates train_Mixtime --lmd 0.1 --sigma 0.02 --noise_reg True --norm False --experiment_root experiments_game_lmd01_sigma002_norm_false


echo lmd:1 sigma: 0.01 norm: false
python run.py --templates train_Mixtime --lmd 1 --sigma 0.01 --noise_reg True --norm False --experiment_root experiments_game_lmd1_sigma001_norm_false


echo lmd:1 sigma: 0.1 norm: true
python run.py --templates train_Mixtime --lmd 1 --sigma 0.1 --noise_reg True --norm False --experiment_root experiments_game_lmd1_sigma01_norm_false

