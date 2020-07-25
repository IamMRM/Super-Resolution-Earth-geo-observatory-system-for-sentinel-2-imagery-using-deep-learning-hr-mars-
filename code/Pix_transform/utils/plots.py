import numpy as np
import matplotlib.pyplot as plt


def plot_result(guide_img, input_img_nearest, output_img, bicubic_img, label_img=None, data_type="rgb", fig_size=(16, 4)):
    cmap = "Spectral"
    guide_img = guide_img.reshape(256,256,3)

    """guide_min = np.percentile(guide_img, 0.05, axis=(0, 1), keepdims=True)  # guide_img.min(axis=(1,2),keepdims=True)
    guide_max = np.percentile(guide_img, 99.95, axis=(0, 1), keepdims=True)  # guide_img.max(axis=(1,2),keepdims=True)
    guide_img = (guide_img - guide_min) / (guide_max - guide_min)
    guide_img = np.clip(guide_img, 0, 1)"""

    if label_img is not None:
        vmin = 0#np.min(label_img)
        vmax = 7#np.max(label_img)

        f, axarr = plt.subplots(1, 5, figsize=fig_size)

        if len(guide_img.shape) > 2:
            axarr[0].imshow(guide_img)
        else:
            axarr[0].imshow(guide_img, cmap="gray")

        axarr[1].imshow(input_img_nearest, vmin=vmin, vmax=vmax, cmap=cmap)

        axarr[2].imshow(label_img, vmin=vmin, vmax=vmax, cmap=cmap)

        axarr[3].imshow(output_img, vmin=vmin, vmax=vmax, cmap=cmap)

        axarr[4].imshow(bicubic_img, vmin=vmin, vmax=vmax, cmap=cmap)
        print(label_img.shape)
        print(output_img.shape)
        titles = ['Guide', 'Source', 'Target',
                  'Predicted Target (MSE {:.3f})'.format(np.mean((label_img - output_img) ** 2)),
                  'Predicted Target (MSE {:.3f})'.format(np.mean((label_img - output_img) ** 2))]
    else:
        vmin = np.min(input_img_nearest)
        vmax = np.max(input_img_nearest)

        f, axarr = plt.subplots(1, 4, figsize=fig_size)
        if len(guide_img.shape) > 2:
            axarr[0].imshow(guide_img)
        else:
            axarr[0].imshow(guide_img, cmap="gray")

        axarr[1].imshow(input_img_nearest, vmin=vmin, vmax=vmax, cmap=cmap)

        axarr[2].imshow(output_img, vmin=vmin, vmax=vmax, cmap=cmap)

        axarr[3].imshow(bicubic_img, vmin=vmin, vmax=vmax, cmap=cmap)

        titles = ['Guide', 'Source', 'Predicted Target',
                  'Bicubic']

    for i, ax in enumerate(axarr):
        ax.set_axis_off()
        ax.set_title(titles[i])

    plt.tight_layout()
    return f, axarr
