from attacker.query import *
from utils import DEVICE

from tqdm import tqdm
import pprint


# Training for Attacker 

def attacker_training(attacker_model, trainloader, testloader, victim_type, num_classes, k=None, verbose=False):
    
    
    # initialize score lists
    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []
    train_loss_func = None
    
    # select optimiser and scheduler
    optimizer = torch.optim.Adam(attacker_model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=config["base_lr"], max_lr=config["max_lr"],
                                                  step_size_up=config["lr_steps"], mode='triangular2', cycle_momentum=False)

    # K_LOGITS_LOGIC - select loss
    if(k != None):
        loss = torch.nn.KLDivLoss(reduction='batchmean')
    else:
        loss = torch.nn.CrossEntropyLoss()
    
    
    # K_LOGITS_LOGIC -need victim
    if(k != None):
        # fetch victim model
        victim_model = fetch_victim_model(args=victim_type)
       
    # calculate epochs from learning cycles
    epochs = int(2*config["lr_cycles"]*config["lr_steps"]/(len(trainloader.dataset)/config["batch_size"]))
    if verbose: print(f'Total epochs to run: {epochs}')
    
    # run epochs
    for epoch in tqdm(range(epochs), disable=verbose):
        if verbose: print(f'\r\tepoch {epoch + 1}')
        
        # train on Train Data
        num_train = 0
        num_correct_train = 0
        attacker_model.train() 
        for (xList, yList) in trainloader:
            # xList, yList = torch.autograd.Variable(xList), torch.autograd.Variable(yList)
            optimizer.zero_grad()

            if torch.cuda.is_available():
                xList = xList.type(torch.cuda.FloatTensor)
                yList = yList.type(torch.cuda.LongTensor)
            attacker_model.to(DEVICE)

            # K_LOGITS_LOGIC
            if(k != None):
                # query victim for logits
                if victim_type["data"] in [CIFAR_10, OOD]:
                    xList_vic = transform_victim_C10(xList)
                else:
                    xList_vic = transform_victim_C100(xList)
                yList = victim_model(xList_vic)
                val, ind = torch.topk(yList, k, dim=1)
                ones = (torch.ones(yList.shape)*float('-inf'))
                if torch.cuda.is_available():
                    ones = ones.type(torch.cuda.FloatTensor)
                yList = ones.scatter_(1, ind, val)
                yList = torch.nn.functional.softmax(yList, dim=1)

            # perform data augmentation on train inputs  
            xList = transform_data_augment(xList)
            
            # get outputs and train model
            with torch.set_grad_enabled(True):
                outputs = attacker_model(xList)
                # K_LOGITS_LOGIC
                if(k != None):
                    # for KL div loss
                    outputs = torch.nn.functional.log_softmax(outputs, dim=1)
                train_loss_func = loss(outputs, yList)
                train_loss_func.backward()
                optimizer.step()
                scheduler.step()
                
            # K_LOGITS_LOGIC
            if(k != None):
                # change k-logits to labels for accuracy calucaltion
                yList = torch.max(yList.data, 1)[1]
                
            # get correct predictions count
            num_train += len(yList)
            predicts = torch.max(outputs.data, 1)[1]
            num_correct_train += (predicts == yList).float().sum()

            if verbose: print('\r\t %d ...' % num_train, end='')

        train_acc.append(num_correct_train / num_train)
        train_loss.append(train_loss_func.data)
        if verbose: print("\r\t- train_acc %.5f train_loss %.5f" % (train_acc[-1], train_loss[-1]))

        # evaluate on Test Data
        num_test = 0
        num_correct_test = 0
        attacker_model.eval()
        for (xList, yList) in testloader:
            if torch.cuda.is_available():
                xList = xList.type(torch.cuda.FloatTensor)
                yList = yList.type(torch.cuda.LongTensor)
            attacker_model.to(DEVICE)
            
            # perform normalization on test inputs  
            xList = transform_normalize(xList)
            
            #  get outputs
            with torch.set_grad_enabled(False):
                outputs = attacker_model(xList)
                # K_LOGITS_LOGIC
                if(k != None):
                    # for KL div loss
                    outputs = torch.nn.functional.log_softmax(outputs, dim=1)
                    yList = torch.nn.functional.one_hot(yList, num_classes=num_classes)
                test_loss_func = loss(outputs, yList)
            
            # K_LOGITS_LOGIC
            if(k != None):
                # change k-logits to labels for accuracy calucaltion
                yList = torch.max(yList.data, 1)[1]
                
            # get correct predictions count
            num_test += len(yList)
            predicts = torch.max(outputs.data, 1)[1]
            num_correct_test += (predicts == yList).float().sum()

        test_acc.append(num_correct_test / num_test)
        test_loss.append(test_loss_func.data)
        if verbose: print("\r\t- test_acc  %.5f test_loss  %.5f" %(test_acc[-1], test_loss[-1]))

    return attacker_model, (train_loss, train_acc, test_loss, test_acc)


# Investigate Attacker training

def investigate(parameters, verbose=False, seed=None):
    """

    :param data_type:
    :type data_type: 
    :return: None
    :rtype:
    """
    
    # set seed before anything
    set_seed(seed)
    
    print('PARAMETERS for investigation:')
    pprint.pprint(parameters)
    results = []
    
    # Iterate through Victim Model & Dataset
    for victim_type in parameters["victim"]:
        print('---------------------------------------------------------------------------')
        # get dataset in dataloader
        trainloader, testloader, outputs = get_dataloader(victim_type["data"])

        # query test data
        querytestloader = query_victim(victim_type, outputs, testloader, len(testloader.dataset), train=False)
       
        # Iterate Through Query Type
        for querytype in parameters["query_type"]:
            # Iterate Through Query Size
            for size in parameters["query_size"]:
                # Iterate Through Logits size
                for k in parameters["k_logits"]:
                    # K_LOGITS_LOGIC
                    if(k != None):
                        # sample train data on querytype.size
                        fo = f'queried/query_{victim_type["data"]}_{victim_type["model_name"]}_traindata_k{k}'
                        querytrainloader = query_type(victim_type, outputs, trainloader, size, fo, querytype)
                    else:
                        # query train data
                        querytrainloader = query_victim(victim_type, outputs, trainloader, size, q_type=querytype)

                    # Iterate through Attacker Model
                    for attacker_type in parameters["attacker"]:
                        print('\n-----------------------------------------------------------------------------')
                        print(f'\tDataset: {victim_type["data"]}')
                        print(f'\tVictim: {victim_type["model_name"]}\tAttacker: {attacker_type}')
                        print(f'\tQuery Type: {querytype}\tQuery Size: {size}\tLogits: {k}')
                        print('-----------------------------------------------------------------------------')

                        # initialize attacker model
                        attacker = get_model(attacker_type, outputs)

                        # train attacker model
                        attacker, attacker_result = attacker_training(attacker, querytrainloader, querytestloader, 
                                                            victim_type, outputs, k=k, verbose=verbose)
                        

                        
                        # save & visualize model inference
                        title = f'A_{attacker_type}_{victim_type["model_name"]}_{victim_type["data"]}_{querytype}_{size}_k{k}'
                        percent = save_visualize(attacker, attacker_result, title)
                        results_dict={"Victim":victim_type["model_name"],
                                        "Dataset":victim_type["data"],
                                        "QueryType": querytype,
                                        "QuerySize": size,
                                        "Queried Logits": k,
                                        "Attacker": attacker_type,
                                        "Train Loss": percent[0],
                                        "Train Accuracy": percent[1],
                                        "Test Loss": percent[2],
                                        "Test Accuracy": percent[3]}

                        pprint.pprint(results_dict)
                        results.append(results_dict)
                        print('-----------------------------------------------------------------------------')
    
    pprint.pprint(results)
    return


