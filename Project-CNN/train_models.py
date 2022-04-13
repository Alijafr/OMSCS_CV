import util
from models import custom_model, VGG16
import os 
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
import argparse
import sys
import pickle


format2_mat_path_train = "dataset/train/format2/train_32x32.mat"
format2__mat_path_test = "dataset/test/format2/test_32x32.mat"
format2_mat_path_extra = "dataset/train/extra_32x32.mat"
format1_mat_path_train = "dataset/train/format1/digitStruct.mat"
format1_mat_path_test = "dataset/test/format1/digitStruct.mat"
format1_path = "dataset/train/format1/"
format1_test_path = "dataset/test/format1/"


def apply_transforms(X,train=True):
    data_transforms = {'train':transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(32),
        transforms.RandomAffine(20),
        transforms.RandomRotation(40),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
    'test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    }
    
    tensor = torch.zeros((X.shape[0],X.shape[3],X.shape[2],X.shape[1]))
    for i in range(len(tensor)):
        if train:
            tensor[i,:] = data_transforms['train'](X[i,:])
        else:
            tensor[i,:] = data_transforms['test'](X[i,:])
        
    
    return tensor

def tensor2numpy_image(tensor):
    
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)#make sure that the values are from 0 to 1 after unnomalization
    return image

def plot_images(imgs, labels, nrows, ncols):
    """ Plot nrows x ncols images
    """
    fig, axes = plt.subplots(nrows, ncols)
    for i, ax in enumerate(axes.flat): 
        if imgs[i].shape == (32, 32, 3):
            ax.imshow(imgs[i])
        else:
            ax.imshow(imgs[i,:,:,0])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(labels[i])

def tensor_imshow(tensor):
    '''
    Helper to visualize tensor images from loaders 
    '''
    image = tensor2numpy_image(tensor)
    #rearrange the height,width and channel
    plt.imshow(image)  # convert from Tensor image       

def plot_hist(train_y,test_y,bins=11):
    fig,axes  = plt.subplots(1, 2,sharex=True,figsize=(10,5))
    fig.suptitle('Dataset Distribution', fontsize=10,fontweight='bold')
    
    axes[0].hist(train_y)
    axes[0].set_title("training data")
    axes[0].set_xlim(1,bins)
    
    axes[1].hist(test_y)
    axes[1].set_title("testing data")
    axes[1].set_xlim(1,bins)
    fig.tight_layout()
    
    plt.savefig("histograms.png")
    
    
def prepare_dataset_torch(transfoms=True):
    #load dataset
    global format2_mat_path_train
    global format2__mat_path_test
    global format2_mat_path_extra
    global format1_path
    global format1_mat_path_train
    global format1_test_path
    global format1_mat_path_test
    
    
    #this is 32x32 images (format2 of SVHN dataset)
    X1_train, Y1_labels, X1_test, Y1_test =  util.preproces_data(format2_mat_path_train,format2__mat_path_test,format2_mat_path_extra) 
    #get negative images from format2
    X2_train ,Y2_lables = util.create_non_digit_dataset(format1_path, format1_mat_path_train,p=1.0)
    X2_test , Y2_test = util.create_non_digit_dataset(format1_test_path, format1_mat_path_test,p=1.0)
    
    #concatenate the dataset together
    X_train = np.concatenate((X1_train,X2_train))
    Y_train = np.concatenate((Y1_labels,Y2_lables))
    
    #shulle training data --> label 10 are all at end now
    shuflled_indices = np.random.permutation(X_train.shape[0])
    X_train = np.take(X_train,shuflled_indices,axis=0)
    Y_train = np.take(Y_train,shuflled_indices,axis=0)
    
    X_test = np.concatenate((X1_test,X2_test[:5000,:,:,:]))
    Y_test = np.concatenate((Y1_test,Y2_test[:5000]))
    
    #save the histograms 
    plot_hist(Y_train,Y_test,bins=11)
    #if transfoms:
    X_train = apply_transforms(X_train,train=transfoms) #if false, the testing transforms will apply
    X_test = apply_transforms(X_test,train=False)
    # else:
    #     X_train = torch.Tensor(X_train)
    #     X_test = torch.Tensor(X_test)
    
    #devide the data into train, valid,and test
    Y_train = torch.Tensor(Y_train)
    Y_test = torch.Tensor(Y_test)
    p=0.9
    X_val = X_train[int(p*len(X_train)):,:]
    Y_val = Y_train[int(p*len(Y_train)):]
    X_train = X_train[:int(p*len(X_train)),:]
    Y_train = Y_train[:int(p*len(Y_train))]
    
    #create torch dataset
    train_dataset = TensorDataset(X_train,Y_train.type(torch.LongTensor))
    val_dataset = TensorDataset(X_val,Y_val.type(torch.LongTensor))
    test_dataset = TensorDataset(X_test,Y_test.type(torch.LongTensor))
    
    #create the dataloaders 
    
    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,num_workers=num_workers)
    test_loader =torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,num_workers=num_workers)
    
    #combine the loaders in a dict
    loaders={'train':train_loader,'valid':valid_loader,'test':test_loader}
    
    return loaders

def save_data_picke():
    #save the data loaders in a pickle to save time
    loaders = prepare_dataset_torch()
    outfile = open('loaders.pickle','wb')
    pickle.dump(loaders, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    outfile.close()

def load_pickle(filename):
    infile = open(filename,'rb')
    pickle_data = pickle.load(infile)
    infile.close()
    return pickle_data
    
    
def get_loss():
    
    ### select loss function
    return nn.CrossEntropyLoss()

def get_optimzer(model,type_ = "SGD"):
    lr = 3e-6
    ### select optimizer
    if type_ == "SGD":
        return torch.optim.SGD(model.parameters(), lr=3e-5, momentum=0.9)
    else:
        return optim.Adam(model.parameters(),lr=lr)
    
def train(n_epochs, loaders, model, save_path,track_train_loss,track_valid_loss,track_train_accuray,track_valid_accuray, valid_loss_min = np.Inf,optimzer_type="SGD"):
    
    optimizer = get_optimzer(model,type_=optimzer_type)
    criterion = get_loss()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        corrects_train = 0
        corrects_valid = 0
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output=model(data)
            # calculate the batch loss
            loss = criterion(output,target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()*data.size(0)
            
            #counts the number of currects labels 
            pred = output.data.max(1, keepdim=True)[1]
            # compare predictions to true label
            corrects_train += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss 
            valid_loss += loss.item()*data.size(0)
            
            #counts the number of currects labels 
            pred = output.data.max(1, keepdim=True)[1]
            # compare predictions to true label
            corrects_valid += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
    
        # calculate average losses
        train_loss = train_loss/len(loaders['train'].dataset)
        valid_loss = valid_loss/len(loaders['valid'].dataset)
        track_train_loss.append(train_loss)
        track_valid_loss.append(valid_loss)
        #calcualate accuarcy 
        track_train_accuray.append(corrects_train/len(loaders['train'].dataset))
        track_valid_accuray.append(corrects_valid/len(loaders['valid'].dataset))
        
        
        

            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss    
    # return trained model
    return model,track_train_loss,track_valid_loss,track_train_accuray,track_valid_accuray,valid_loss_min

def test(loaders, model):
    
    criterion = get_loss()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        #print(" {} out {}".format(batch_idx, len(loaders['test'].dataset)))
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))
   
def plot_result(x1,x2,figure_name,title, loss=True):
    plt.plot(x1,label="training")
    plt.plot(x2,label="validation")
    plt.legend()
    plt.grid()
    plt.xlabel('Epoch')
    plt.title(title)
    if loss:
        plt.ylabel("loss")
        #plt.title("Loss for {}".format(figure_name))
        plt.savefig('loss_{}.png'.format(figure_name)) 
    else:
        plt.ylabel("Accuracy")
        #plt.title("Accuracy for {}".format(figure_name))
        plt.savefig('Accuracy_{}.png'.format(figure_name)) 
    
    plt.close()
    #plt.ylim(0, 7) # consistent scale
    #plt.show()    
# if __name__ == "__main__":
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_type",default="custom_model")
#     parser.add_argument("--out_weights_file",default="custom_model.pt",required=True)
#     parser.add_argument("--pretained",required=False) #only for VGG16 
#     parser.add_argument("--pickle_file",required=False) #only for VGG16 
#     args = parser.parse_args()
#     if args.model_type == "custom_model":    
#         model = custom_model(in_channels=3,num_classes= 11)
#         #save_weight_name = "custom_model.pt"
#         figure_name = "custom_model_training"
#     elif args.model_type == "VGG16":
#         if args.pretained == "True":
#             model = VGG16(pretrained=True,in_channels=3,num_classes=11)
#             #save_weight_name = "pre_trained_vgg.pt"
#             figure_name = "vgg_pretrained"
#         elif args.pretained == "False":
#             model = VGG16(pretrained=False,in_channels=3,num_classes=11)
#             #save_weight_name = "vgg_retrained.pt"
#             figure_name = "vgg_retrained"
#         else:
#             print("wrong input, should be eithre True or False (case-sensative)")
#             print("going with pretrained VGG16")
#             model = VGG16(pretrained=True,in_channels=3,num_classes=11)
#             #save_weight_name = "pre_trained_vgg.pt"
#             figure_name = "vgg_pretrained"
#     else:
#         print("wrong input for model_type, should be either custom_model or VGG16")
#         sys.exit()
        
#     save_weight_name =  args.out_weights_file       
#     print("training {} command received..... ".format(args.model_type))
#     #get the loader 
#     if args.pickle_file != None:
#         print("loading the pickle file")
#         loaders = load_pickle(args.pickle_file)
#     else:
#         print("no picke file, loading data and data loaders.......")
#         loaders = prepare_dataset_torch()
    
#     #train model 
    
#     track_train_loss=[]
#     track_valid_loss=[]
#     track_train_accuray = []
#     track_valid_accuray = []
#     min_val_loss=np.Inf
#     #train the model
#     print("training started.......")
#     model,track_train_loss,track_valid_loss,track_train_accuray,track_valid_accuray,valid_loss_min = train(7, loaders, model, save_weight_name, track_train_loss, track_valid_loss,track_train_accuray,track_valid_accuray,min_val_loss)
    
#     print("testing started.......")
#     print("loading the saved weights")
#     model.load_state_dict(torch.load(save_weight_name))
#     #test accuracy
#     test(loaders, model)
#     print("ploting.......")
#     #plot and save result 
#     plot_result(track_train_loss, track_valid_loss,figure_name,"Loss for VGG16 [pre-trained]", loss=True)
#     plot_result(track_train_accuray,track_valid_accuray,figure_name,"Accuracy for VGG16 [pre-trained]",loss=False)
    
    
if __name__ == "__main__":
    loaders = load_pickle("loaders_with_extra.pickle")
    # models = []
    # models.append(custom_model(in_channels=3,num_classes= 11))
    # models.append(VGG16(pretrained=True,in_channels=3,num_classes=11))
    # models.append(VGG16(pretrained=False,in_channels=3,num_classes=11))
    # weight_names =  ["custom_model_extra2.pt","vgg_pretrained_extra2.pt","vgg_retrained_extra2.pt"]      
    # figure_name  = ["custom_model_extra2","vgg_pretrained_extra2","vgg_retrained_extra2"]
    # titles_loss = ["Loss for custom model","Loss for VGG16 [pre-trained]", "Loss for VGG16 [re-trained]"]
    # titles_acc = ["Accuracy for custom model","Accuracy for VGG16 [pre-trained]", "Accuracy for VGG16 [re-trained]"]
    
    # for i in range(len(models)):
    #     #train model 
        
    #     track_train_loss=[]
    #     track_valid_loss=[]
    #     track_train_accuray = []
    #     track_valid_accuray = []
    #     min_val_loss=np.Inf
    #     #train the model
    #     print("training started.......")
    #     models[i],track_train_loss,track_valid_loss,track_train_accuray,track_valid_accuray,valid_loss_min = train(15, loaders, models[i], weight_names[i], track_train_loss, track_valid_loss,track_train_accuray,track_valid_accuray,min_val_loss,optimzer_type="ADAM")
        
    #     print("testing started.......")
    #     print("loading the saved weights")
    #     torch.save(models[i].state_dict(), "last_{}.pt".format(str(i)))
    #     test(loaders, models[i])
    #     models[i].load_state_dict(torch.load(weight_names[i]))
    #     #test accuracy
    #     test(loaders, models[i])
    #     print("ploting.......")
    #     #plot and save result 
    #     plot_result(track_train_loss, track_valid_loss,figure_name[i],titles_loss[i], loss=True)
    #     plot_result(track_train_accuray,track_valid_accuray,figure_name[i],titles_acc[i],loss=False)
        
    models = []
    models.append(custom_model(in_channels=3,num_classes= 11))
    models.append(VGG16(pretrained=True,in_channels=3,num_classes=11))
    models.append(VGG16(pretrained=False,in_channels=3,num_classes=11))
    weight_names =  ["custom_model_extra3.pt","vgg_pretrained_extra3.pt","vgg_retrained_extra3.pt"]      
    figure_name  = ["custom_model_extra3","vgg_pretrained_extra3","vgg_retrained_extra3"]
    titles_loss = ["Loss for custom model","Loss for VGG16 [pre-trained]", "Loss for VGG16 [re-trained]"]
    titles_acc = ["Accuracy for custom model","Accuracy for VGG16 [pre-trained]", "Accuracy for VGG16 [re-trained]"]
    
    for i in range(len(models)):
        #train model 
        
        track_train_loss=[]
        track_valid_loss=[]
        track_train_accuray = []
        track_valid_accuray = []
        min_val_loss=np.Inf
        #train the model
        print("training started.......")
        models[i],track_train_loss,track_valid_loss,track_train_accuray,track_valid_accuray,valid_loss_min = train(40, loaders, models[i], weight_names[i], track_train_loss, track_valid_loss,track_train_accuray,track_valid_accuray,min_val_loss)
        
        print("testing started.......")
        print("loading the saved weights")
        torch.save(models[i].state_dict(), "last2_{}.pt".format(str(i)))
        test(loaders, models[i])
        models[i].load_state_dict(torch.load(weight_names[i]))
        #test accuracy
        test(loaders, models[i])
        print("ploting.......")
        #plot and save result 
        plot_result(track_train_loss, track_valid_loss,figure_name[i],titles_loss[i], loss=True)
        plot_result(track_train_accuray,track_valid_accuray,figure_name[i],titles_acc[i],loss=False)






