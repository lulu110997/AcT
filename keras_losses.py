import matplotlib.pyplot as plt
import os
import pickle_wrapper as _pw
k = "/home/louis/Data/Fernandez_HAR/AcT_keras_results/micro_trained_with_losses/losses/"
pt = "/home/louis/Data/Fernandez_HAR/AcT_pt/trying_to_reach_keras_results/micro_with_pret/plots"
for split in range(1, 4):
    for fold in range(10):
        k_training_loss = f"training_loss_s{split}_f{fold}_micro.pickle"
        k_validation_loss = f"validation_loss_s{split}_f{fold}_micro.pickle"
        k_tl = _pw.open_pickle(os.path.join(k, k_training_loss))
        k_vl = _pw.open_pickle(os.path.join(k, k_validation_loss))
        pt_training_loss = f"s_{split}_f_{fold}_train_loss.pickle"
        pt_validation_loss = f"s_{split}_f_{fold}_test_loss.pickle"
        pt_tl = _pw.open_pickle(os.path.join(pt, pt_training_loss))
        pt_vl = _pw.open_pickle(os.path.join(pt, pt_validation_loss))
        fig, ax = plt.subplots()
        ax.set_title(f'Loss for split {split} and fold {fold}')
        ax.plot(k_tl, label='keras training loss')
        ax.plot(k_vl, label='keras validation_loss')
        ax.plot(pt_tl, label='pt training loss')
        ax.plot(pt_vl, label='pt validation_loss')
        plt.legend()
        plt.show()