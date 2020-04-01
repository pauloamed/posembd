import random
import sys
import tqdm

'''
do_policy: aplly generator do_policy
    - emilia: uses all samples from all datasets, without shuffle
    - visconde: uses all samples from all datasets, with shuffle

get_batches: extract batches from (train, val or test) subsets from datasets
'''


def do_policy(policy, datasets, batchSize, samples):
    seed = random.randrange(sys.maxsize)

    batches, numBatches = [], []

    for i in range(len(datasets)):
        numBatches.append(len(samples[i][0])//batchSize)
        samples[i] = (samples[i][0][0:numBatches[-1] * batchSize],
                           samples[i][1][0:numBatches[-1] * batchSize])


    for i in range(len(datasets)):
        for ii in range(numBatches[i]):
            start = ii * batchSize
            end = (ii+1) * batchSize

            if(samples[i][0][start:end] == []):
                continue

            inputsBatch = samples[i][0][start:end]
            targetsBatch = samples[i][1][start:end]

            batches.append((inputsBatch, targetsBatch, datasets[i].name))

    if policy == "emilia":
        pass
    elif policy == "visconde":
        random.Random(seed).shuffle(batches)
    else:
        pass

    return batches

def get_batches(datasets, tvt, batchSize=1, policy="emilia", numBatches = -1):
    samples = []

    if tvt == "train":
        datasets = [d for d in datasets if d.useTrain]
        totalLen = sum([dataset.sentCountTrain for dataset in datasets])
        samples = [(d.trainInput, d.trainTarget) for d in datasets]
    elif tvt == 'val':
        datasets = [d for d in datasets if d.useVal]
        totalLen = sum([dataset.sentCountVal for dataset in datasets])
        samples = [(d.valInput, d.valTarget) for d in datasets]
    elif tvt == 'test':
        totalLen = sum([dataset.sentCountTest for dataset in datasets])
        samples = [(d.testInput, d.testTarget) for d in datasets]



    batches = do_policy(policy, datasets, batchSize, samples)

    if numBatches != -1:
        numBatches = min(len(batches), numBatches)
        batches = batches[:numBatches]


    desc = "{}: batchSize={}, policy={}".format(tvt, batchSize, policy)
    for b in tqdm.tqdm(batches, desc=desc, file=sys.stdout):
        yield b
