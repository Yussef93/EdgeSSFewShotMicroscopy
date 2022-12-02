import os
import pickle
import shutil


def extractFewShotTargetSelection(target_dir,num_shots,datasets):
    #num_shots = ['1-shot','3-shot','5-shot','7-shot','10-shot']
    #datasets = ['B5','B39','TNBC','EM','ssTEM']
    #num_shots = ['5-shot']
    #datasets = ['TNBC']
    for num_selection in range(1,11):
        if not os.path.exists(target_dir+'Target/Selection_'+str(num_selection)+'/FinetuneSamples/') \
            and not os.path.exists(target_dir+'Target/Selection_'+str(num_selection)+'/TestSamples/'):
            os.makedirs(target_dir+'Target/Selection_'+str(num_selection)+'/FinetuneSamples/')
            os.makedirs(target_dir+'Target/Selection_'+str(num_selection)+'/TestSamples/')
        for shot in num_shots:
            for dataset in datasets:
                if not os.path.exists(target_dir+'Target/Selection_' + str(num_selection) +
                          '/FinetuneSamples/'+str(shot)+'-shot/'+dataset+'/Image/') \
                    and not os.path.exists(target_dir+'Target/Selection_' + str(num_selection) +
                          '/TestSamples/' + str(shot) + '-shot/' + dataset + '/Image/'):

                    os.makedirs(target_dir+'Target/Selection_' + str(num_selection) +
                              '/FinetuneSamples/'+str(shot)+'-shot/'+dataset+'/Image/')
                    os.makedirs(target_dir+'Target/Selection_' + str(num_selection) +
                              '/FinetuneSamples/' + str(shot) + '-shot/' + dataset + '/Groundtruth/')
                    os.makedirs(target_dir+'Target/Selection_' + str(num_selection) +
                              '/TestSamples/' + str(shot) + '-shot/' + dataset + '/Image/')
                    os.makedirs(target_dir+'Target/Selection_' + str(num_selection) +
                              '/TestSamples/' + str(shot) + '-shot/' + dataset + '/Groundtruth/')

    f = open("./selections.pkl", "rb")
    selections = pickle.load(f)
    f.close()
    main_dir = target_dir+'Target/'
    dataset_dir = './Datasets/Raw/'
    for selection in selections:
        print("Extracting "+selection)
        finetune_dir = main_dir+selection+'/FinetuneSamples/'
        test_dir = main_dir+selection+'/TestSamples/'
        for shot in num_shots:#selections[selection]['FinetuneSamples']:
            shot = str(shot)+'-shot'
            for dataset in selections[selection]['FinetuneSamples'][shot]:
                if dataset == 'TNBC':
                    ft = {'images':[],'groundtruth':[]}
                    test = {'images':[],'groundtruth':[]}
                    raw = {'images':[],'groundtruth':[]}

                    if dataset != 'TNBC':


                        raw['images'] += os.listdir(dataset_dir+dataset+'/Image/')
                        raw['groundtruth'] += os.listdir(dataset_dir+dataset+'/Groundtruth/')
                    else:
                        ground_truth_folders = sorted(
                            [folder + '/' for folder in os.listdir(dataset_dir+dataset+'/Groundtruth/') if folder[0] == 'G'])
                        image_folders = sorted(
                            [folder + '/' for folder in os.listdir(dataset_dir+dataset+'/Image/') if folder[0] == 'S'])

                        for folder in ground_truth_folders:
                            raw['groundtruth'] += sorted([dataset_dir+dataset+ '/Groundtruth/'+ folder + f for f in
                                                          os.listdir(dataset_dir+dataset+ '/Groundtruth/'+ folder) if
                                                          f[0] != '.'])
                        for folder in image_folders:
                            raw['images'] += sorted(
                                [dataset_dir+dataset+ '/Image/' + folder + f for f in os.listdir(dataset_dir+dataset+ '/Image/'+ folder) if
                                 f[0] != '.'])
                    for image in sorted(raw['images']):
                        if os.path.basename(image) in sorted(selections[selection]['FinetuneSamples'][shot][dataset]['Image']):
                            ft['images'].append(image)

                        else:
                            test['images'].append(image)

                    for gt in sorted(raw['groundtruth']):
                        if os.path.basename(gt) in sorted(selections[selection]['FinetuneSamples'][shot][dataset]['Groundtruth']):
                            ft['groundtruth'].append(gt)

                        else:
                            test['groundtruth'].append(gt)

                    for image,gt in zip (ft['images'],ft['groundtruth']):
                        if dataset !='TNBC':
                            shutil.copy(dataset_dir+dataset+'/Image/'+image, finetune_dir+shot+'/'+dataset+'/Image/')
                            shutil.copy(dataset_dir+dataset+'/Groundtruth/'+gt, finetune_dir + shot + '/' + dataset + '/Groundtruth/')
                        else:
                            shutil.copy(image,finetune_dir + shot + '/' + dataset + '/Image/')
                            shutil.copy(gt,finetune_dir + shot + '/' + dataset + '/Groundtruth/')

                    for image,gt in zip (test['images'],test['groundtruth']):
                        if dataset != 'TNBC':
                            shutil.copy(dataset_dir+dataset+'/Image/'+image, test_dir+shot+'/'+dataset+'/Image/')
                            shutil.copy(dataset_dir+dataset+'/Groundtruth/'+gt, test_dir + shot + '/' + dataset + '/Groundtruth/')
                        else:
                            shutil.copy(image, test_dir + shot + '/' + dataset + '/Image/')
                            shutil.copy( gt,
                                        test_dir + shot + '/' + dataset + '/Groundtruth/')

    print("Target selections extracted!\n")
