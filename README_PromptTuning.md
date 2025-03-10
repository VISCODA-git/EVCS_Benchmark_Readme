The prompt tuning part is adapted from the code implemented here:
https://github.com/CMZJIRCV/mmdetection/tree/dev-3.x-prompttuning/projects/GroundingdinoPT 

* Config file construction

    To build the config file defined under projects/GroundingdinoPT/config. The following parts needes to be changed:
    - data root and language model name

    ```
    data_root = '/data/sequences/coco/'
    lang_model_name = 'bert-base-uncased'
    ```
    - num_class and dataset loader for training, valiation and test under seperate data loaders


* Training

    ```
    python tools/train.py config_file --resume grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth 
    ```
    Note: the pretrained files are downloaded from https://github.com/open-mmlab/mmdetection/tree/main/configs/mm_grounding_dino (mm_grounding_dino)

* Test

    To test the model, the default test tools is used with 
    ```
    python tools/test.py CONFIG_FILE MODEL_FILE 
    ```
    
* Model Slicing

    After training, the model must be sliced to seperate the learned prompt and the saved model weight. An example usage is:

    ```
     python projects/GroundingdinoPT/model_split_to_prompt_pth.py --weight_path work_dirs/grounding_dino_swinT_EVCS_oneclass_prompt/epoch_39.pth --real_name_list 'electric vehicle charging station' --save_path outputs/
    ```

* Apply the model for single image
    by specifying inputs (image path), model (config) texts (classes), prompt_pth (the learned prompt), weights (the weight file of pretrained grouding dino) 

    ```
    python projects/GroundingdinoPT/single_image_inference.py --inputs data/EVCSDataset_VISCODA_V1.1_additional_data/additional/site_46/image_additional000460.jpg --model projects/GroundingdinoPT/config/grounding_dino_swinT_EVCS_oneclass_prompt.py  --texts 'car. traffic sign' --prompt_pth '../../outputs/electric vehicle charging station.pth' --weights grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth
    ```
    The output are detection .json and visualization of single images
    .json file contains ['labels', 'scores', 'bboxes'] that can be extracted for further processing

    "texts" can be any class names seperated by '.'
    <!--
    Question: How do they join the text embeddings to get the final prediction?
    

    BaseModel -> BaseDetector -> DetectionTransformer -> DeformableDETR -> DINO -> GroudingDINO forward()
    .predict() is called

    for languge model, the prompt length of learned embedding is 10 and it is stiched with the input text features (like for other classes)
    -->


