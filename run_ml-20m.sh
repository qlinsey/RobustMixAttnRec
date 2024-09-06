echo no-noise-reg
python run.py --templates train_Mixtime --noise_reg False --experiment_root experiments_ml-20m_no_noise
echo 'done Mixtime'

echo lmd:0.01 sigma: 0.2 norm: true
python run.py --templates train_Mixtime --lmd 0.01 --sigma 0.2 --noise_reg True --norm True --experiment_root experiments_ml-20m_lmd001_sigma02_norm_true
echo 'done Mixtime'


echo lmd:0.01 sigma: 0.2 norm: false
python run.py --templates train_Mixtime --lmd 0.01 --sigma 0.2 --noise_reg True --norm False --experiment_root experiments_ml-20m_lmd001_sigma02_norm_false
echo 'done Mixtime'

echo lmd:0.1 sigma: 0.2 norm: true
python run.py --templates train_Mixtime --lmd 0.1 --sigma 0.2 --noise_reg True --norm True --experiment_root experiments_ml-20m_lmd01_sigma02_norm_true
echo 'done Mixtime'


echo lmd:0.1 sigma: 0.2 norm: false
python run.py --templates train_Mixtime --lmd 0.1 --sigma 0.2 --noise_reg True --norm False --experiment_root experiments_ml-20m_lmd01_sigma02_norm_false
echo 'done Mixtime'

echo lmd:0.01 sigma: 0.02 norm: true
python run.py --templates train_Mixtime --lmd 0.01 --sigma 0.02 --noise_reg True --norm True --experiment_root experiments_ml-20m_lmd001_sigma002_norm_true


echo lmd:0.1 sigma: 0.02 norm: false
python run.py --templates train_Mixtime --lmd 0.1 --sigma 0.02 --noise_reg True --norm False --experiment_root experiments_ml-20m_lmd01_sigma002_norm_false


echo lmd:1 sigma: 0.01 norm: false
python run.py --templates train_Mixtime --lmd 1 --sigma 0.01 --noise_reg True --norm False --experiment_root experiments_ml-20m_lmd1_sigma001_norm_false


echo lmd:1 sigma: 0.1 norm: true
python run.py --templates train_Mixtime --lmd 1 --sigma 0.1 --noise_reg True --norm False --experiment_root experiments_ml-20m_lmd1_sigma01_norm_false

