class DatasetCatalog(object):
    dataset_attrs = {
        'BScan3D_Train': {
            'id': 'BScan3D',
            'img_dir': 'data/BScanSeg/V1/train/Ori_img',
            'enhanced_dir': 'data/BScanSeg/V1/train/Ori_enhanced_img',
            'label_f_dir': 'data/BScanSeg/V1/train/Ori_label_f',
            'label_s_dir': 'data/BScanSeg/V1/train/Ori_label_s',

            'seq_img_dir': '',
            'seq_enhanced_dir': '',
            'seq_label_f_dir': '',
            'seq_label_s_dir': '',

            'split': 'train'
        },
        'BScan3D_United_Train': {
            'id': 'BScan3D',
            'img_dir': 'data/BScanSeg/V1/train/United_img',
            'enhanced_dir': 'data/BScanSeg/V1/train/United_enhanced_img',
            'label_f_dir': 'data/BScanSeg/V1/train/United_label_f',
            'label_s_dir': 'data/BScanSeg/V1/train/United_label_s',

            'seq_img_dir': '',
            'seq_enhanced_dir': '',
            'seq_label_f_dir': '',
            'seq_label_s_dir': '',

            'split': 'train'
        },
        'BScan3D_Val': {
            'id': 'BScan3D',
            'img_dir': 'data/BScanSeg/V1/val/Ori_img',
            'enhanced_dir': 'data/BScanSeg/V1/val/Ori_enhanced_img',
            'label_f_dir': 'data/BScanSeg/V1/val/Ori_label_f',
            'label_s_dir': 'data/BScanSeg/V1/val/Ori_label_s',
            'split': 'test'
        },
        'BScan3D_United_Val': {
            'id': 'BScan3D',
            'img_dir': 'data/BScanSeg/V1/val/United_img',
            'enhanced_dir': 'data/BScanSeg/V1/val/United_enhanced_img',
            'label_f_dir': 'data/BScanSeg/V1/val/United_label_f',
            'label_s_dir': 'data/BScanSeg/V1/val/United_label_s',
            'split': 'test'
        },

        'BScan3D_United_Train_V2': {
            'id': 'BScan3D',
            'img_dir': 'data/BScanSeg/V2/train/United_img',
            'enhanced_dir': 'data/BScanSeg/V2/train/United_enhanced_img',
            'label_f_dir': 'data/BScanSeg/V2/train/United_label_f',
            'label_s_dir': 'data/BScanSeg/V2/train/United_label_s',

            'seq_img_dir': '',
            'seq_enhanced_dir': '',
            'seq_label_f_dir': '',
            'seq_label_s_dir': '',

            'split': 'train'
        },
        'BScan3D_United_Val_V2': {
            'id': 'BScan3D',
            'img_dir': 'data/BScanSeg/V2/val/United_img',
            'enhanced_dir': 'data/BScanSeg/V2/val/United_enhanced_img',
            'label_f_dir': 'data/BScanSeg/V2/val/United_label_f',
            'label_s_dir': 'data/BScanSeg/V2/val/United_label_s',
            'split': 'test'
        },

        'BScan3D_United_Train_V3': {
            'id': 'BScan3D',
            'img_dir': 'data/BScanSeg/V3/train/UNT_IMG',
            'enhanced_dir': 'data/BScanSeg/V3/train/UNE_IMG',
            'label_f_dir': 'data/BScanSeg/V3/train/UNT_LABEL',

            'enh_dir': 'data/BScanSeg/V3/train/ENH_IMG',
            'ori_dir': 'data/BScanSeg/V3/train/ORI_IMG',
            'label_o_dir': 'data/BScanSeg/V3/train/ORI_LABEL',

            'mini_img': '',
            'mini_label': '',

            'split': 'train'
        },
        'BScan3D_United_Val_V3': {
            'id': 'BScan3D',
            'img_dir': 'data/BScanSeg/V3/val/UNT_IMG',
            'enhanced_dir': 'data/BScanSeg/V3/val/UNE_IMG',
            'label_f_dir': 'data/BScanSeg/V3/val/UNT_LABEL',

            'enh_dir': 'data/BScanSeg/V3/val/ENH_IMG',
            'ori_dir': 'data/BScanSeg/V3/val/ORI_IMG',
            'label_o_dir': 'data/BScanSeg/V3/val/ORI_LABEL',

            'mini_img': '',
            'mini_label': '',

            'split': 'test'
        },

        'BScan3D_United_Train_V4': {
            'id': 'BScan3D',
            'img_dir': 'data/BScanSeg/V4/train/SHAPE_IMG',
            'label_dir': 'data/BScanSeg/V4/train/SHAPE_LABEL',
            'mini_img': '',
            'mini_label': '',
            'split': 'train'
        },
        'BScan3D_United_Val_V4': {
            'id': 'BScan3D',
            'img_dir': 'data/BScanSeg/V4/val/SHAPE_IMG',
            'label_dir': 'data/BScanSeg/V4/val/SHAPE_LABEL',
            'mini_img': '',
            'mini_label': '',
            'split': 'test'
        },

        'BScan3D_United_Train_V5': {
            'id': 'BScan3D',
            'img_dir': 'data/BScanSeg/V5/train/PAD_IMG',
            'enhanced_dir': 'data/BScanSeg/V5/train/PADE_IMG',
            'label_f_dir': 'data/BScanSeg/V5/train/PAD_LABEL',

            'mini_img': '',
            'mini_label': '',
            'split': 'train'
        },
        'BScan3D_United_Val_V5': {
            'id': 'BScan3D',
            'img_dir': 'data/BScanSeg/V5/val/PAD_IMG',
            'enhanced_dir': 'data/BScanSeg/V5/val/PADE_IMG',
            'label_f_dir': 'data/BScanSeg/V5/val/PAD_LABEL',

            'mini_img': '',
            'mini_label': '',
            'split': 'test',
        },
        # #####################---------------------------##################### #
        'OrganSeg_Train_V0': {
            'id': 'MultiOrganSeg',
            'img_dir': 'data/MultiOrganSeg/TrainingImg',
            'label_dir': 'data/MultiOrganSeg/TrainingMask',
            'split': 'train'
        },
        'OrganSeg_Test_V0': {
            'id': 'MultiOrganSeg',
            'img_dir': 'data/MultiOrganSeg/ValidationImage',
            'label_dir': 'data/MultiOrganSeg/ValidationMask',
            'split': 'test'
        },

    }

    @staticmethod
    def get(name):
        attrs = DatasetCatalog.dataset_attrs[name]
        return attrs.copy()
