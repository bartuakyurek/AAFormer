from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np

img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]
mean = np.array(img_mean)
std = np.array(img_std)

transform_img = transforms.Compose(
    [
        transforms.Normalize(-mean/std, 1/std),
        transforms.ToPILImage(),
    ]
 )

transform_mask = transforms.Compose(
  [
      #transforms.Normalize(-mean/std, 1/std),
      transforms.ToPILImage(),
  ]
)


def image_plot(query_img, supp_imgs, supp_masks, preds):

    preds = preds.cpu()

    fig = plt.figure()
    i = 0
    for q in range(query_img.shape[0]):
        plt.subplot(query_img.shape[1],query_img.shape[0],i+1)
        plt.imshow(transform_img(query_img[q,:,:,:]))
        plt.suptitle("query_img", y=0.7)
        plt.axis('off')
        i += 1
    plt.show()
    
    
    i = 0
    for s in range(supp_imgs.shape[1]):
        for supp_img in supp_imgs[:,s,:,:]:
            plt.subplot(supp_imgs.shape[1],supp_imgs.shape[0],i+1)
            plt.imshow(transform_img(supp_img))
            plt.suptitle("supp_img", y=0.7)
            plt.axis('off')
            i += 1
    plt.show()

    i = 0
    for s in range(supp_masks.shape[1]):
        for supp_mask in supp_masks[:,s,:,:]:
            plt.subplot(supp_masks.shape[1],supp_masks.shape[0],i+1)
            plt.imshow(transform_mask(supp_mask))
            plt.suptitle("supp_masks", y=0.7)
            plt.axis('off')
            i += 1
    plt.show()


    i = 0
    for pred in preds:
        plt.subplot(1,preds.shape[0],i+1)
        plt.imshow(pred)
        plt.suptitle("preds", y=0.7)
        plt.axis('off')
        i += 1
    plt.show()

#plt.savefig("test.png", bbox_inches='tight')