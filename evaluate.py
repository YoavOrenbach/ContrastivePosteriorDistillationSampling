import pickle

import torch
from utils.utils import load_default_model
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import os
from PIL import Image
import torchvision.transforms as transforms
from glob import glob
from tabulate import tabulate
import matplotlib.pyplot as plt


def evaluate_models():
    """ Evaluates all model configurations (DDS/CDS/PDS/CPDS) on a cats dataset using 3 prompts:
        cat->dog, cat->cow, cat->pig"""

    device = torch.device(f'cuda:{0}') if torch.cuda.is_available() else torch.device('cpu')
    stable = load_default_model()

    stable = stable.to(device)
    generator = torch.Generator(device).manual_seed(0)

    imgs_path = 'sample/cats_small'
    save_path = 'results/cats'

    if not os.path.isdir(imgs_path):
        return
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    img_files = sorted(glob(os.path.join(imgs_path, '*.jpg')))
    input_prompt = "a cat"
    target_prompts = ["a dog", "a cow", "a pig"]

    dds_clip_scores, dds_lpips_scores = [], []
    cds_clip_scores, cds_lpips_scores = [], []
    pds_clip_scores, pds_lpips_scores = [], []
    cpds_clip_scores, cpds_lpips_scores = [], []

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    clip_score_fn = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')

    for target_prompt in target_prompts:
        target_save_path = save_path + '/' + target_prompt
        if not os.path.isdir(target_save_path):
            os.makedirs(target_save_path)

        for img_file in img_files:
            image = Image.open(img_file)
            input_image_tensor = transform(image).unsqueeze(0)

            for config in ["dds", "cds", "pds", "cpds"]:
                result = stable(
                    img_path=img_file,
                    prompt=input_prompt,
                    trg_prompt=target_prompt,
                    num_inference_steps=200,
                    generator=generator,
                    n_patches=256,
                    patch_size=[1,2],
                    save_path=save_path,
                    config=config
                )

                # Save result
                result.save(os.path.join(target_save_path, config+'_'+os.path.basename(img_file)))

                result_image_tensor = transform(result).unsqueeze(0)

                clip_score = clip_score_fn(result_image_tensor, target_prompt) / 100
                lpips_score = lpips_fn(input_image_tensor, result_image_tensor)

                print(f'---- {config} scores ----')
                print(f'{config} clip score: {clip_score.item():}')
                print(f'{config} lpips score: {lpips_score.item()}')
                if config == "dds":
                    dds_clip_scores.append(clip_score.item())
                    dds_lpips_scores.append(lpips_score.item())
                elif config == "cds":
                    cds_clip_scores.append(clip_score.item())
                    cds_lpips_scores.append(lpips_score.item())
                elif config == "pds":
                    pds_clip_scores.append(clip_score.item())
                    pds_lpips_scores.append(lpips_score.item())
                else:
                    cpds_clip_scores.append(clip_score.item())
                    cpds_lpips_scores.append(lpips_score.item())


        if target_prompt == "a dog":
            dog_dds_clip_avg = sum(dds_clip_scores) / len(dds_clip_scores)
            dog_dds_lpips_avg = sum(dds_lpips_scores) / len(dds_lpips_scores)
            dog_cds_clip_avg = sum(cds_clip_scores) / len(cds_clip_scores)
            dog_cds_lpips_avg = sum(cds_lpips_scores) / len(cds_lpips_scores)
            dog_pds_clip_avg = sum(pds_clip_scores) / len(pds_clip_scores)
            dog_pds_lpips_avg = sum(pds_lpips_scores) / len(pds_lpips_scores)
            dog_cpds_clip_avg = sum(cpds_clip_scores) / len(cpds_clip_scores)
            dog_cpds_lpips_avg = sum(cpds_lpips_scores) / len(cpds_lpips_scores)
        elif target_prompt == "a cow":
            cow_dds_clip_avg = sum(dds_clip_scores) / len(dds_clip_scores)
            cow_dds_lpips_avg = sum(dds_lpips_scores) / len(dds_lpips_scores)
            cow_cds_clip_avg = sum(cds_clip_scores) / len(cds_clip_scores)
            cow_cds_lpips_avg = sum(cds_lpips_scores) / len(cds_lpips_scores)
            cow_pds_clip_avg = sum(pds_clip_scores) / len(pds_clip_scores)
            cow_pds_lpips_avg = sum(pds_lpips_scores) / len(pds_lpips_scores)
            cow_cpds_clip_avg = sum(cpds_clip_scores) / len(cpds_clip_scores)
            cow_cpds_lpips_avg = sum(cpds_lpips_scores) / len(cpds_lpips_scores)
        else:
            pig_dds_clip_avg = sum(dds_clip_scores) / len(dds_clip_scores)
            pig_dds_lpips_avg = sum(dds_lpips_scores) / len(dds_lpips_scores)
            pig_cds_clip_avg = sum(cds_clip_scores) / len(cds_clip_scores)
            pig_cds_lpips_avg = sum(cds_lpips_scores) / len(cds_lpips_scores)
            pig_pds_clip_avg = sum(pds_clip_scores) / len(pds_clip_scores)
            pig_pds_lpips_avg = sum(pds_lpips_scores) / len(pds_lpips_scores)
            pig_cpds_clip_avg = sum(cpds_clip_scores) / len(cpds_clip_scores)
            pig_cpds_lpips_avg = sum(cpds_lpips_scores) / len(cpds_lpips_scores)

        cds_clip_scores.clear()
        cds_lpips_scores.clear()
        pds_clip_scores.clear()
        pds_lpips_scores.clear()
        cpds_clip_scores.clear()
        cpds_lpips_scores.clear()

    # Display results in a table
    table = [['Method', 'cat->dog CLIP Score', 'cat->dog LPIPS Score', 'cat->cow CLIP Score', 'cat->cow LPIPS Score', 'cat->pig CLIP Score', 'cat->pig LPIPS Score'],
             ['DDS', dog_dds_clip_avg, dog_dds_lpips_avg, cow_dds_clip_avg, cow_dds_lpips_avg, pig_dds_clip_avg, pig_dds_lpips_avg],
             ['CDS', dog_cds_clip_avg, dog_cds_lpips_avg, cow_cds_clip_avg, cow_cds_lpips_avg, pig_cds_clip_avg, pig_cds_lpips_avg],
             ['PDS', dog_pds_clip_avg, dog_pds_lpips_avg, cow_pds_clip_avg, cow_pds_lpips_avg, pig_pds_clip_avg, pig_pds_lpips_avg],
             ['CPDS', dog_cpds_clip_avg, dog_cpds_lpips_avg, cow_cpds_clip_avg, cow_cpds_lpips_avg, pig_cpds_clip_avg, pig_cpds_lpips_avg]]
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))


def evaluate_patch_selection():
    """ Evaluates patch selection techniques"""
    device = torch.device(f'cuda:{0}') if torch.cuda.is_available() else torch.device('cpu')
    stable = load_default_model()

    stable = stable.to(device)
    generator = torch.Generator(device).manual_seed(0)

    img_file = "sample/cat1.png"
    save_path = 'results/selector'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    input_prompt = "a cat"
    target_prompt = "a pig"

    clip_scores, lpips_scores = [], []
    random_accuracies, meta_accuracies = [], []

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    clip_score_fn = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')

    image = Image.open(img_file)
    input_image_tensor = transform(image).unsqueeze(0)

    for i, patch_selector in enumerate(["random", "meta"]):
        result = stable(
            img_path=img_file,
            prompt=input_prompt,
            trg_prompt=target_prompt,
            num_inference_steps=200,
            generator=generator,
            n_patches=256,
            patch_size=[1, 2],
            save_path=save_path,
            config="cds",
            patch_selector=patch_selector
        )

        # Save result
        result.save(os.path.join(save_path, patch_selector + '_' + os.path.basename(img_file)))

        result_image_tensor = transform(result).unsqueeze(0)

        clip_score = clip_score_fn(result_image_tensor, target_prompt) / 100
        lpips_score = lpips_fn(input_image_tensor, result_image_tensor)

        print(f'---- {patch_selector} scores ----')
        print(f'{patch_selector} clip score: {clip_score.item():}')
        print(f'{patch_selector} lpips score: {lpips_score.item()}')
        clip_scores.append(clip_score.item())
        lpips_scores.append(lpips_score.item())

        with open("results/selector/accuracies", "rb") as fp:
            if patch_selector == "random":
                random_accuracies = pickle.load(fp)
            elif patch_selector == "meta":
                meta_accuracies = pickle.load(fp)

    table = [['Method', 'CLIP Score', 'LPIPS Score'],
             ['Random', clip_scores[0], lpips_scores[0]],
             ['Meta-Learner', clip_scores[1], lpips_scores[1]]]
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

    plt.figure(figsize=(10, 6))
    plt.plot(random_accuracies, label='Random selector')
    plt.plot(meta_accuracies, label='Meta-Learner selector')
    plt.title("Patch selector accuracies")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy %")
    plt.legend()
    plt.grid()
    plt.show()


def evaluate_sample():

    device = torch.device(f'cuda:{0}') if torch.cuda.is_available() else torch.device('cpu')
    stable = load_default_model()

    stable = stable.to(device)
    generator = torch.Generator(device).manual_seed(0)

    #img_file = 'sample/cat1.png'
    #save_path = 'results/samples/cat1'
    img_file = 'sample/candle.jpg'
    save_path = 'results/samples/candle_lighter'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    input_prompt = "a candle"
    target_prompt = "a lighter"

    for config in ["dds", "cds", "pds", "cpds"]:
        result = stable(
            img_path=img_file,
            prompt=input_prompt,
            trg_prompt=target_prompt,
            num_inference_steps=200,
            generator=generator,
            n_patches=256,
            patch_size=[1,2],
            save_path=save_path,
            config=config
        )

        # Save result
        result.save(os.path.join(save_path, config+'_'+os.path.basename(img_file)))


if __name__ == '__main__':
    #evaluate_models()
    #evaluate_patch_selection()
    evaluate_sample()
