import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import time
import random
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
plt.rcParams.update({'font.size': 22})

def plot_latent_space(vae, data, n_synthetic_data, min_ls, max_ls, latent_dim, min=0, max=1):
  fig1 = plt.figure(figsize=(20,10))
  gs = gridspec.GridSpec(3, 4)

  def inverseScalerThetis(xscaled, xmin, xmax, min, max):
    scale = (max - min) / (xmax - xmin)
    xInv = (xscaled/scale) - (min/scale) + xmin
    return xInv
  start = time.time()
  size_gen_data = n_synthetic_data#data.shape[0]
  noise = np.random.normal(size=(size_gen_data, latent_dim))
  predicted_data = vae.decoder.predict(noise)
  predicted_data = inverseScalerThetis(predicted_data, min_ls, max_ls, min, max)
  real_data = inverseScalerThetis(data, min_ls, max_ls, min, max)
  #predicted_data = (predicted_data*stdData) + meanData
  #real_data = (predicted_data*stdData) + meanData
  predicted_data = np.exp(predicted_data)
  real_data = np.exp(real_data)

  predicted_data = predicted_data - 1
  real_data = real_data - 1
  end = time.time() - start
  print(end)
  plt.subplot(4, 2, 1)
  plt.scatter(real_data[:,0], real_data[:, 1], c=real_data[:, -1], cmap='jet')
  #plt.ylim(0, 0.025)
  #plt.xlim(0, 0.4)
  plt.title('Flow pattern map (ground truth)')
  plt.xlabel('2Qc')
  plt.ylabel('Qd')
  plt.subplot(4,2,2)
  plt.scatter(predicted_data[:, 0], predicted_data[:, 1], c=predicted_data[:,-1], cmap='jet')
  #plt.ylim(0,0.025)
  #plt.xlim(0,0.4)
  plt.xlabel('2Qc')
  plt.ylabel('Qd')
  plt.title('Flow pattern map (generated)')
  plt.subplot(4,2,3)
  random_pred_data = random.sample(sorted(set(predicted_data[:, -1])), data.shape[0])
  plt.scatter(np.sort(real_data[:,-1]), np.sort(random_pred_data))
  ident = [0.0, 0.2]
  plt.plot(ident,ident, color='r')
  plt.ylim(0,0.2)
  plt.title('Droplet diameter (mm) (ground truth)')
  plt.xlabel('Sorted samples')
  plt.ylabel('ms')
  plt.subplot(4,2,4)
  plt.plot(np.sort(random_pred_data))
  plt.plot(np.sort(real_data[:, -1]))
  MAPE = np.mean(np.abs(np.sort(random_pred_data)- np.sort(real_data[:, -1]))/np.sort(real_data[:, -1]))
  print(MAPE)
  plt.ylim(0,0.2)
  plt.title('Droplet diameter (mm) (generated)')
  plt.xlabel('Sorted samples')
  plt.ylabel('ms')
  plt.subplot(4,2,5)
  plt.scatter(real_data[:, 0], real_data[:, 1], c=real_data[:,2], cmap='jet')
  #plt.ylim(0,0.025)
  #plt.xlim(0,0.4)
  plt.xlabel('2Qc')
  plt.ylabel('Qd')
  plt.subplot(4,2,6)
  plt.scatter(predicted_data[:, 0], predicted_data[:, 1], c=predicted_data[:,2], cmap='jet')
  #plt.ylim(0,0.025)
  #plt.xlim(0,0.4)
  plt.xlabel('2Qc')
  plt.ylabel('Qd')
  plt.tight_layout()
  fig2 = plt.figure(figsize=(20,10))
  #np.save('generated_data_4features', predicted_data)
  #variable_names = ['Qc', 'Qd', '$\gamma$', '$\phi/\phi_{CMC}','Droplet diameter']
  variable_names = ['$Q_{c}$', '$Q_{d}$', '$\gamma$', '$\phi/\phi_{CMC}$', 'd']
  #Mutual Information

  from scipy.stats import entropy, ks_2samp
  from scipy.stats import probplot
  flag = 1
  for i in range(5):
    if i < 4:
      plt.subplot(3,2,i+1)
    else:
      ax3 = plt.subplot(gs[2:, 1:3])
    sns.kdeplot(real_data[:, i], fill=True, alpha = .3, palette="crest")
    sns.kdeplot(random.sample(sorted(set(predicted_data[:, i])), data.shape[0]), fill = True, alpha =.3, palette="crest")
    plt.title(variable_names[i])
    #plt.xlabel('Experimental')
    #plt.ylabel('Synthetic')
    if flag == 1:
      plt.legend(['Experimental', 'Synthetic'])
      flag = 0
    plt.gca().set_xlim(left=0)
  plt.tight_layout()
  return fig1, fig2, MAPE, end