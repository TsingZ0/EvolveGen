import os
import pandas as pd
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import ssl
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ssl._create_default_https_context = ssl._create_stdlib_context

# mean=[0.485, 0.456, 0.406]
# std=[0.229, 0.224, 0.225]
# normalize = transforms.Normalize(mean=mean, std=std)
# inv_normalize = transforms.Normalize(
#     mean=[-m/s for m, s in zip(mean, std)],
#     std=[1/s for s in std]
# )


def select_data(any_set, start, end):
    any_loader = DataLoader(any_set, len(any_set), shuffle=False)
    data, label = next(iter(any_loader))
    any_set_data = data.cpu().detach().numpy()
    any_set_label = label.cpu().detach().numpy()

    label_ids = set(any_set_label)
    any_set_data_used = None
    any_set_label_used = None
    for label_id in label_ids:
        if any_set_data_used is None:
            any_set_data_used = any_set_data[any_set_label == label_id][start:end]
            any_set_label_used = any_set_label[any_set_label == label_id][start:end]
        else:
            any_set_data_used = np.append(
                any_set_data_used, 
                any_set_data[any_set_label == label_id][start:end], axis=0
            )
            any_set_label_used = np.append(
                any_set_label_used, 
                any_set_label[any_set_label == label_id][start:end], axis=0
            )
    any_set = list(zip(any_set_data_used, any_set_label_used))
    return any_set


def get_real_data(args):
    real_exist, test_exist = False, False
    real_dataset_dir = os.path.join(args.dataset_dir, f'real/{args.real_volume_per_label}')
    test_dataset_dir = os.path.join(args.dataset_dir, 'test')
    real_file_path = os.path.join(real_dataset_dir, args.client_dataset + '.pt')
    test_file_path = os.path.join(test_dataset_dir, args.client_dataset + '.pt')
    label_file_path = os.path.join(test_dataset_dir, args.client_dataset + '-label_names.pt')
    task_file_path = os.path.join(test_dataset_dir, args.client_dataset + '-domain.pt')
    if args.real_volume_per_label == 0:
        real_exist = True
    else:
        if not os.path.exists(real_dataset_dir):
            os.makedirs(real_dataset_dir)
        elif os.path.exists(real_file_path):
            real_exist = True
    if not os.path.exists(test_dataset_dir):
        os.makedirs(test_dataset_dir)
    elif os.path.exists(test_file_path):
        test_exist = True

    def get_transform(img_size):
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        return transform

    if not real_exist or not test_exist:
        if args.client_dataset == 'Cifar10':
            any_set = torchvision.datasets.CIFAR10(
                root=os.path.join(args.dataset_dir, 'rawdata'), 
                train=True, 
                download=True, 
                transform=transforms.ToTensor()
            )
            any_loader = DataLoader(any_set)
            H = next(iter(any_loader))[0].shape[-2]
            W = next(iter(any_loader))[0].shape[-1]
            args.img_size = min(min(H, W), args.image_max_size)
            real_set = torchvision.datasets.CIFAR10(
                root=os.path.join(args.dataset_dir, 'rawdata'), 
                train=True, 
                download=True, 
                transform=get_transform(args.img_size)
            )
            test_set = torchvision.datasets.CIFAR10(
                root=os.path.join(args.dataset_dir, 'rawdata'), 
                train=False, 
                download=True, 
                transform=get_transform(args.img_size)
            )
            domain = 'common'
            label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 
                        'horse', 'ship', 'truck']
        elif args.client_dataset == 'Cifar100':
            any_set = torchvision.datasets.CIFAR100(
                root=os.path.join(args.dataset_dir, 'rawdata'), 
                train=True, 
                download=True, 
                transform=transforms.ToTensor()
            )
            any_loader = DataLoader(any_set)
            H = next(iter(any_loader))[0].shape[-2]
            W = next(iter(any_loader))[0].shape[-1]
            args.img_size = min(min(H, W), args.image_max_size)
            real_set = torchvision.datasets.CIFAR100(
                root=os.path.join(args.dataset_dir, 'rawdata'), 
                train=True, 
                download=True, 
                transform=get_transform(args.img_size)
            )
            test_set = torchvision.datasets.CIFAR100(
                root=os.path.join(args.dataset_dir, 'rawdata'), 
                train=False, 
                download=True, 
                transform=get_transform(args.img_size)
            )
            domain = 'common'
            label_names = ['beaver', 'dolphin', 'otter', 'seal', 'whale', 
                            'aquarium fish', 'flatfish', 'ray', 'shark', 'trout', 
                            'orchids', 'poppies', 'roses', 'sunflowers', 'tulips', 
                            'bottles', 'bowls', 'cans', 'cups', 'plates', 
                            'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers', 
                            'clock', 'computer keyboard', 'lamp', 'telephone', 'television', 
                            'bed', 'chair', 'couch', 'table', 'wardrobe', 
                            'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach', 
                            'bear', 'leopard', 'lion', 'tiger', 'wolf', 
                            'bridge', 'castle', 'house', 'road', 'skyscraper', 
                            'cloud', 'forest', 'mountain', 'plain', 'sea', 
                            'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo', 
                            'fox', 'porcupine', 'possum', 'raccoon', 'skunk', 
                            'crab', 'lobster', 'snail', 'spider', 'worm', 
                            'baby', 'boy', 'girl', 'man', 'woman', 
                            'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle', 
                            'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel', 
                            'maple', 'oak', 'palm', 'pine', 'willow', 
                            'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train', 
                            'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor']
        elif args.client_dataset == 'Flowers102':
            any_set = torchvision.datasets.Flowers102(
                root=os.path.join(args.dataset_dir, 'rawdata'), 
                split='train', 
                download=True, 
                transform=transforms.ToTensor()
            )
            any_loader = DataLoader(any_set)
            H = next(iter(any_loader))[0].shape[-2]
            W = next(iter(any_loader))[0].shape[-1]
            args.img_size = min(min(H, W), args.image_max_size)
            real_set = torchvision.datasets.Flowers102(
                root=os.path.join(args.dataset_dir, 'rawdata'), 
                split='train', 
                download=True, 
                transform=get_transform(args.img_size)
            )
            test_set = torchvision.datasets.Flowers102(
                root=os.path.join(args.dataset_dir, 'rawdata'), 
                split='test', 
                download=True, 
                transform=get_transform(args.img_size)
            )
            domain = 'flower'
            label_names = ['alpine sea holly', 'anthurium', 'artichoke', 'azalea', 
                            'ball moss', 'balloon flower', 'barbeton daisy', 'bearded iris', 
                            'bee balm', 'bird of paradise', 'bishop of llandaff', 
                            'black-eyed susan', 'blackberry lily', 'blanket flower', 
                            'bolero deep blue', 'bougainvillea', 'bromelia', 'king protea', 
                            'lenten rose', 'lotus', 'love in the mist', 'magnolia', 
                            'mallow', 'marireal', 'mexican aster', 'mexican petunia', 
                            'monkshood', 'moon orchid', 'morning glory', 'orange dahlia', 
                            'osteospermum', 'oxeye daisy', 'passion flower', 'pelargonium', 
                            'buttercup', 'californian poppy', 'camellia', 'canna lily', 
                            'canterbury bells', 'cape flower', 'carnation', 'cautleya spicata', 
                            'clematis', 'colt\'s foot', 'columbine', 'common dandelion', 
                            'corn poppy', 'cyclamen ', 'daffodil', 'desert-rose', 
                            'english marireal', 'peruvian lily', 'petunia', 'pincushion flower', 
                            'pink primrose', 'pink-yellow dahlia?', 'poinsettia', 'primula', 
                            'prince of wales feathers', 'purple coneflower', 'red ginger', 
                            'rose', 'ruby-lipped cattleya', 'siam tulip', 'silverbush', 
                            'snapdragon', 'spear thistle', 'spring crocus', 'fire lily', 
                            'foxglove', 'frangipani', 'fritillary', 'garden phlox', 'gaura', 
                            'gazania', 'geranium', 'giant white arum lily', 'globe thistle', 
                            'globe-flower', 'grape hyacinth', 'great masterwort', 
                            'hard-leaved pocket orchid', 'hibiscus', 'hippeastrum ', 
                            'japanese anemone', 'stemless gentian', 'sunflower', 'sweet pea', 
                            'sweet william', 'sword lily', 'thorn apple', 'tiger lily', 
                            'toad lily', 'tree mallow', 'tree poppy', 'trumpet creeper', 
                            'wallflower', 'water lily', 'watercress', 'wild pansy', 
                            'windflower', 'yellow iris']
        elif args.client_dataset == 'DTD':
            any_set = torchvision.datasets.DTD(
                root=os.path.join(args.dataset_dir, 'rawdata'), 
                split='train', 
                download=True, 
                transform=transforms.ToTensor()
            )
            any_loader = DataLoader(any_set)
            H = next(iter(any_loader))[0].shape[-2]
            W = next(iter(any_loader))[0].shape[-1]
            args.img_size = min(min(H, W), args.image_max_size)
            real_set = torchvision.datasets.DTD(
                root=os.path.join(args.dataset_dir, 'rawdata'), 
                split='train', 
                download=True, 
                transform=get_transform(args.img_size)
            )
            test_set = torchvision.datasets.DTD(
                root=os.path.join(args.dataset_dir, 'rawdata'), 
                split='test', 
                download=True, 
                transform=get_transform(args.img_size)
            )
            domain = 'describable textures'
            label_names = ['banded texture', 'blotchy texture', 'braided texture', 'bubbly texture', 
                        'bumpy texture', 'chequered texture', 'cobwebbed texture', 'cracked texture', 
                        'crosshatched texture', 'crystalline texture', 'dotted texture', 'fibrous texture', 
                        'flecked texture', 'freckled texture', 'frilly texture', 'gauzy texture', 
                        'grid texture', 'grooved texture', 'honeycombed texture', 'interlaced texture', 
                        'knitted texture', 'lacelike texture', 'lined texture', 'marbled texture', 
                        'matted texture', 'meshed texture', 'paisley texture', 'perforated texture', 
                        'pitted texture', 'pleated texture', 'polka-dotted texture', 'porous texture', 
                        'potholed texture', 'scaly texture', 'smeared texture', 'spiralled texture', 
                        'sprinkled texture', 'stained texture', 'stratified texture', 'striped texture', 
                        'studded texture', 'swirly texture', 'veined texture', 'waffled texture', 
                        'woven texture', 'wrinkled texture', 'zigzagged texture']
        elif args.client_dataset == 'EuroSAT':
            any_set = torchvision.datasets.EuroSAT(
                root=os.path.join(args.dataset_dir, 'rawdata'), 
                download=True, 
                transform=transforms.ToTensor()
            )
            any_loader = DataLoader(any_set)
            H = next(iter(any_loader))[0].shape[-2]
            W = next(iter(any_loader))[0].shape[-1]
            args.img_size = min(min(H, W), args.image_max_size)
            full_set = torchvision.datasets.EuroSAT(
                root=os.path.join(args.dataset_dir, 'rawdata'), 
                download=True, 
                transform=get_transform(args.img_size)
            )
            domain = 'satellite'
            label_names = ['Annual Crop Land', 'Forest', 'Herbaceous Vegetation Land', 'Highway or Road', 
                        'Industrial Building',  'Pasture Land', 'Permanent Crop Land', 'Residential Building', 
                        'River', 'Sea or Lake']
            full_len_per_label = len(full_set) // len(label_names)
            test_index = int(full_len_per_label * args.test_ratio)
            test_set = select_data(full_set, 0, test_index)
            real_set = select_data(full_set, test_index, full_len_per_label)
        elif args.client_dataset == 'SUN397':
            any_set = torchvision.datasets.SUN397(
                root=os.path.join(args.dataset_dir, 'rawdata'), 
                download=True, 
                transform=transforms.ToTensor()
            )
            any_loader = DataLoader(any_set)
            H = next(iter(any_loader))[0].shape[-2]
            W = next(iter(any_loader))[0].shape[-1]
            args.img_size = min(min(H, W), args.image_max_size)
            full_set = torchvision.datasets.SUN397(
                root=os.path.join(args.dataset_dir, 'rawdata'), 
                download=True, 
                transform=get_transform(args.img_size)
            )
            domain = 'scene'
            label_names = ['abbey', 'airplane cabin', 'airport terminal', 'alley', 'amphitheater', 
                        'amusement arcade', 'amusement park', 'anechoic chamber', 
                        'apartment building, outdoor', 'apse, indoor', 'aquarium', 
                        'aqueduct', 'arch', 'archive', 'arrival gate, outdoor', 'art gallery', 
                        'art school', 'art studio', 'assembly line', 'athletic field, outdoor', 
                        'atrium, public', 'attic', 'auditorium', 'auto factory', 'badlands', 
                        'badminton court, indoor', 'baggage claim', 'bakery, shop', 'balcony, exterior', 
                        'balcony, interior', 'ball pit', 'ballroom', 'bamboo forest', 'banquet hall', 
                        'bar', 'barn', 'barndoor', 'baseball field', 'basement', 'basilica', 
                        'basketball court, outdoor', 'bathroom', 'batters box', 'bayou', 'bazaar, indoor', 
                        'bazaar, outdoor', 'beach', 'beauty salon', 'bedroom', 'berth', 'biology laboratory', 
                        'bistro, indoor', 'boardwalk', 'boat deck', 'boathouse', 'bookstore', 
                        'booth, indoor', 'botanical garden', 'bow window, indoor', 'bow window, outdoor', 
                        'bowling alley', 'boxing ring', 'brewery, indoor', 'bridge', 'building facade', 
                        'bullring', 'burial chamber', 'bus interior', 'butchers shop', 'butte', 
                        'cabin, outdoor', 'cafeteria', 'campsite', 'campus', 'canal, natural', 
                        'canal, urban', 'candy store', 'canyon', 'car interior, backseat', 
                        'car interior, frontseat', 'carrousel', 'casino, indoor', 'castle', 
                        'catacomb', 'cathedral, indoor', 'cathedral, outdoor', 'cavern, indoor', 
                        'cemetery', 'chalet', 'cheese factory', 'chemistry lab', 'chicken coop, indoor', 
                        'chicken coop, outdoor', 'childs room', 'church, indoor', 'church, outdoor', 
                        'classroom', 'clean room', 'cliff', 'cloister, indoor', 'closet', 'clothing store', 
                        'coast', 'cockpit', 'coffee shop', 'computer room', 'conference center', 
                        'conference room', 'construction site', 'control room', 'control tower, outdoor', 
                        'corn field', 'corral', 'corridor', 'cottage garden', 'courthouse', 'courtroom', 
                        'courtyard', 'covered bridge, exterior', 'creek', 'crevasse', 'crosswalk', 
                        'cubicle, office', 'dam', 'delicatessen', 'dentists office', 'desert, sand', 
                        'desert, vegetation', 'diner, indoor', 'diner, outdoor', 'dinette, home', 
                        'dinette, vehicle', 'dining car', 'dining room', 'discotheque', 'dock', 
                        'doorway, outdoor', 'dorm room', 'driveway', 'driving range, outdoor', 'drugstore', 
                        'electrical substation', 'elevator shaft', 'elevator, door', 'elevator, interior', 
                        'engine room', 'escalator, indoor', 'excavation', 'factory, indoor', 'fairway', 
                        'fastfood restaurant', 'field, cultivated', 'field, wild', 'fire escape', 
                        'fire station', 'firing range, indoor', 'fishpond', 'florist shop, indoor', 
                        'food court', 'forest path', 'forest road', 'forest, broadleaf', 'forest, needleleaf', 
                        'formal garden', 'fountain', 'galley', 'game room', 'garage, indoor', 'garbage dump', 
                        'gas station', 'gazebo, exterior', 'general store, indoor', 'general store, outdoor', 
                        'gift shop', 'golf course', 'greenhouse, indoor', 'greenhouse, outdoor', 
                        'gymnasium, indoor', 'hangar, indoor', 'hangar, outdoor', 'harbor', 'hayfield', 
                        'heliport', 'herb garden', 'highway', 'hill', 'home office', 'hospital', 
                        'hospital room', 'hot spring', 'hot tub, outdoor', 'hotel room', 'hotel, outdoor', 
                        'house', 'hunting lodge, outdoor', 'ice cream parlor', 'ice floe', 'ice shelf', 
                        'ice skating rink, indoor', 'ice skating rink, outdoor', 'iceberg', 'igloo', 
                        'industrial area', 'inn, outdoor', 'islet', 'jacuzzi, indoor', 'jail cell', 
                        'jail, indoor', 'jewelry shop', 'kasbah', 'kennel, indoor', 'kennel, outdoor', 
                        'kindergarden classroom', 'kitchen', 'kitchenette', 'labyrinth, outdoor', 
                        'lake, natural', 'landfill', 'landing deck', 'laundromat', 'lecture room', 
                        'library, indoor', 'library, outdoor', 'lido deck, outdoor', 'lift bridge', 
                        'lighthouse', 'limousine interior', 'living room', 'lobby', 'lock chamber', 
                        'locker room', 'mansion', 'manufactured home', 'market, indoor', 'market, outdoor', 
                        'marsh', 'martial arts gym', 'mausoleum', 'medina', 'moat, water', 
                        'monastery, outdoor', 'mosque, indoor', 'mosque, outdoor', 'motel', 'mountain', 
                        'mountain snowy', 'movie theater, indoor', 'museum, indoor', 'music store', 
                        'music studio', 'nuclear power plant, outdoor', 'nursery', 'oast house', 
                        'observatory, outdoor', 'ocean', 'office', 'office building', 'oil refinery, outdoor', 
                        'oilrig', 'operating room', 'orchard', 'outhouse, outdoor', 'pagoda', 'palace', 
                        'pantry', 'park', 'parking garage, indoor', 'parking garage, outdoor', 'parking lot', 
                        'parlor', 'pasture', 'patio', 'pavilion', 'pharmacy', 'phone booth', 
                        'physics laboratory', 'picnic area', 'pilothouse, indoor', 'planetarium, outdoor', 
                        'playground', 'playroom', 'plaza', 'podium, indoor', 'podium, outdoor', 'pond', 
                        'poolroom, establishment', 'poolroom, home', 'power plant, outdoor', 'promenade deck', 
                        'pub, indoor', 'pulpit', 'putting green', 'racecourse', 'raceway', 'raft', 
                        'railroad track', 'rainforest', 'reception', 'recreation room', 
                        'residential neighborhood', 'restaurant', 'restaurant kitchen', 'restaurant patio', 
                        'rice paddy', 'riding arena', 'river', 'rock arch', 'rope bridge', 'ruin', 
                        'runway', 'sandbar', 'sandbox', 'sauna', 'schoolhouse', 'sea cliff', 'server room', 
                        'shed', 'shoe shop', 'shopfront', 'shopping mall, indoor', 'shower', 'skatepark', 
                        'ski lodge', 'ski resort', 'ski slope', 'sky', 'skyscraper', 'slum', 'snowfield', 
                        'squash court', 'stable', 'stadium, baseball', 'stadium, football', 'stage, indoor', 
                        'staircase', 'street', 'subway interior', 'subway station, platform', 'supermarket', 
                        'sushi bar', 'swamp', 'swimming pool, indoor', 'swimming pool, outdoor', 
                        'synagogue, indoor', 'synagogue, outdoor', 'television studio', 'temple, east asia', 
                        'temple, south asia', 'tennis court, indoor', 'tennis court, outdoor', 
                        'tent, outdoor', 'theater, indoor procenium', 'theater, indoor seats', 'thriftshop', 
                        'throne room', 'ticket booth', 'toll plaza', 'topiary garden', 'tower', 'toyshop', 
                        'track, outdoor', 'train railway', 'train station, platform', 'tree farm', 
                        'tree house', 'trench', 'underwater, coral reef', 'utility room', 'valley', 
                        'van interior', 'vegetable garden', 'veranda', 'veterinarians office', 'viaduct', 
                        'videostore', 'village', 'vineyard', 'volcano', 'volleyball court, indoor', 
                        'volleyball court, outdoor', 'waiting room', 'warehouse, indoor', 'water tower', 
                        'waterfall, block', 'waterfall, fan', 'waterfall, plunge', 'watering hole', 
                        'wave', 'wet bar', 'wheat field', 'wind farm', 'windmill', 
                        'wine cellar, barrel storage', 'wine cellar, bottle storage', 
                        'wrestling ring, indoor', 'yard', 'youth hostel']
            full_len_per_label = len(full_set) // len(label_names)
            test_index = int(full_len_per_label * args.test_ratio)
            test_set = select_data(full_set, 0, test_index)
            real_set = select_data(full_set, test_index, full_len_per_label)
        elif args.client_dataset == 'Camelyon17':
            from wilds import get_dataset
            dataset = get_dataset(
                dataset='camelyon17', 
                root_dir=os.path.join(args.dataset_dir, 'rawdata'), 
                download=True, 
            )
            transform=transforms.ToTensor()
            real_set = [(transform(x), y) for x, y, _ in dataset.get_subset('train')]
            test_set = [(transform(x), y) for x, y, _ in dataset.get_subset('test')]
            H = test_set[0][0].shape[-2]
            W = test_set[0][0].shape[-1]
            args.img_size = min(min(H, W), args.image_max_size)
            domain = 'histological lymph node section'
            label_names = ['', 'breast cancer with a tumor tissue']
        elif args.client_dataset == 'kvasir':
            # https://datasets.simula.no/kvasir/
            data_dir = 'dataset/rawdata/kvasir-dataset-v2/'
            label_names = ['esophagitis', 'polyps', 'ulcerative-colitis']
            file_names = []
            labels = []
            for dir in os.listdir(data_dir):
                if dir in label_names:
                    label = label_names.index(dir)
                    for file_name in os.listdir(os.path.join(data_dir, dir)):
                        file_names.append(os.path.join(dir, file_name))
                        labels.append(label)
            df = pd.DataFrame({'file_name': file_names, 'class': labels})

            dataset = ImageDataset(df, data_dir, transforms.ToTensor())
            any_loader = DataLoader(dataset)
            H = next(iter(any_loader))[0].shape[-2]
            W = next(iter(any_loader))[0].shape[-1]
            args.img_size = min(min(H, W), args.image_max_size)

            transform = transforms.Compose(
                [transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()])
            dataset = ImageDataset(df, data_dir, transform)
            full_len_per_label = len(dataset) // len(label_names)
            test_index = int(full_len_per_label * args.test_ratio)
            test_set = select_data(dataset, 0, test_index)
            real_set = select_data(dataset, test_index, full_len_per_label)
            domain = 'pathological damage in mucosa of gastrointestinal tract'
        elif args.client_dataset == 'COVIDx':
            # https://www.kaggle.com/datasets/andyczhao/covidx-cxr2
            data_dir = 'dataset/rawdata/COVIDx/'
            val_df = pd.read_csv(data_dir + 'val.txt', sep=" ", header=None)
            val_df.columns=['patient_id', 'file_name', 'class', 'data_source']
            val_df.drop(columns=['patient_id', 'data_source'])
            val_df['class'] = val_df['class'] == 'positive'
            val_df['class'] = val_df['class'].astype(int)

            test_df = pd.read_csv(data_dir + 'test.txt', sep=" ", header=None)
            test_df.columns=['patient_id', 'file_name', 'class', 'data_source']
            test_df.drop(columns=['patient_id', 'data_source'])
            test_df['class'] = test_df['class'] == 'positive'
            test_df['class'] = test_df['class'].astype(int)

            any_set = ImageDataset(val_df, data_dir+'val/', transforms.ToTensor())
            any_loader = DataLoader(any_set)
            H = next(iter(any_loader))[0].shape[-2]
            W = next(iter(any_loader))[0].shape[-1]
            args.img_size = min(min(H, W), args.image_max_size)

            transform = transforms.Compose(
                [transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()])
            real_set = ImageDataset(val_df, data_dir+'val/', transform)
            real_loader = DataLoader(real_set)
            real_set = [(x[0], y[0]) for x, y in real_loader]
            test_set = ImageDataset(test_df, data_dir+'test/', transform)
            test_loader = DataLoader(test_set)
            test_set = [(x[0], y[0]) for x, y in test_loader]
            domain = 'chest radiography (X-ray)'
            label_names = ['', 'COVID-19 pneumonia']
        elif args.client_dataset == 'PrivateCat':
            # https://www.kaggle.com/datasets/fjxmlzn/cat-cookie-doudou
            data_dir = 'dataset/rawdata/PrivateCat/'
            dir_names = ['cookie', 'doudou']
            file_names = []
            labels = []
            for dir in os.listdir(data_dir):
                if dir in dir_names:
                    label = dir_names.index(dir)
                    for file_name in os.listdir(os.path.join(data_dir, dir)):
                        file_names.append(os.path.join(dir, file_name))
                        labels.append(label)
            df = pd.DataFrame({'file_name': file_names, 'class': labels})

            dataset = ImageDataset(df, data_dir, transforms.ToTensor())
            any_loader = DataLoader(dataset)
            H = next(iter(any_loader))[0].shape[-2]
            W = next(iter(any_loader))[0].shape[-1]
            args.img_size = min(min(H, W), args.image_max_size)

            transform = transforms.Compose(
                [transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()])
            dataset = ImageDataset(df, data_dir, transform)
            label_names = ['', '_']
            full_len_per_label = len(dataset) // len(label_names)
            test_index = int(full_len_per_label * args.test_ratio)
            test_set = select_data(dataset, 0, test_index)
            real_set = select_data(dataset, test_index, full_len_per_label)
            domain = 'ragdoll cat'
        elif args.client_dataset == 'MVTecADLeather':
            # https://www.mvtec.com/company/research/datasets/mvtec-ad
            data_dir = 'dataset/rawdata/MVTecADLeather/'
            dir_names = ['good', 'cut', 'glue']
            # only 19 images in each class
            file_names = []
            labels = []
            for dir in os.listdir(data_dir):
                if dir in dir_names:
                    label = dir_names.index(dir)
                    for file_name in os.listdir(os.path.join(data_dir, dir)):
                        file_names.append(os.path.join(dir, file_name))
                        labels.append(label)
            df = pd.DataFrame({'file_name': file_names, 'class': labels})

            dataset = ImageDataset(df, data_dir, transforms.ToTensor())
            any_loader = DataLoader(dataset)
            H = next(iter(any_loader))[0].shape[-2]
            W = next(iter(any_loader))[0].shape[-1]
            args.img_size = min(min(H, W), args.image_max_size)

            transform = transforms.Compose(
                [transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()])
            dataset = ImageDataset(df, data_dir, transform)
            label_names = ['', 'cut defect', 'droplet defect']
            full_len_per_label = len(dataset) // len(label_names)
            test_index = int(full_len_per_label * args.test_ratio)
            test_set = select_data(dataset, 0, test_index)
            real_set = select_data(dataset, test_index, full_len_per_label)
            domain = 'leather texture'
        else:
            raise NotImplemented
            
        real_set = select_data(real_set, 0, args.real_volume_per_label)
        print(f'Real and test datasets created.')
        if not os.path.exists(real_file_path):
            torch.save(real_set, real_file_path)
        if not os.path.exists(test_file_path):
            torch.save(test_set, test_file_path)
        if not os.path.exists(label_file_path):
            torch.save(label_names, label_file_path)
        if not os.path.exists(task_file_path):
            torch.save(domain, task_file_path)
    else:
        print('Real and test datasets already exist.')

    test_set = torch.load(test_file_path)
    if args.real_volume_per_label == 0:
        print(f'Test set size: {len(test_set)}.')
    else:
        real_set = torch.load(real_file_path)
        print(f'Real set size: {len(real_set)}, test set size: {len(test_set)}.')

    label_names = torch.load(label_file_path)
    domain = torch.load(task_file_path)
    args.label_names = label_names
    args.num_labels = len(label_names)
    args.domain = domain
    print(f'Labels: {args.label_names}')
    print(f'Number of labels: {args.num_labels}')
    print(f'Client domain: {args.domain}')
    H = test_set[0][0].shape[-2]
    W = test_set[0][0].shape[-1]
    args.img_size = min(H, W)
    print(f'Client image size: {H}x{W}')


def preprocess_image(args, image_path):
    # Load image
    img = Image.open(image_path)

    # Resize image
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])

    # Apply transformations
    img_tensor = transform(img)

    return img_tensor


class ImageDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing file names
            image_folder (str): Path to the folder containing the images
            transform (callable, optional): Optional transform to be applied to the image
        """
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get the file name from the DataFrame
        img_name = self.dataframe.iloc[idx]['file_name']
        img_label = self.dataframe.iloc[idx]['class']
        img_path = os.path.join(self.image_folder, img_name)
        
        # Load the image using PIL
        image = Image.open(img_path).convert('RGB')  # Ensure RGB if not grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_label