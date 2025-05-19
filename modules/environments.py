import os

LANG = "中文"

SCENE = "儿童短剧"

ACTION = "现在有一个一分钟左右的短视频大纲需求"

LASTSHOTSUBTITLE = 'The subtitle for the previous shot is:'

FIRSTSHOTWITHOUTCONTEXT = "This is the first shot, there is no previous context"

from tools import read_text

def getRolesInfoGlobal():
    role_cache  = f"{os.getcwd()}/outputs/roles"
    os.makedirs(role_cache,exist_ok=True)
    roles = [read_text(f"{role_cache}/{role_dir}/desc.txt").split("\n") for role_dir in os.listdir(role_cache) if os.path.exists(f"{role_cache}/{role_dir}/lora.safetensors")]
    roles_info = "\n".join([f"Role\nid:{role[0]}\ndesc:{role[1]}\nstyle:{role[2]}" for role in roles])
    return roles_info


def read_storyboard_prompt():
    storyboard_prompt = f"""
    Although this instructions are given in English, your thought process must be in {LANG}.

    You are a chief director specializing in producing videos about {SCENE}. Your focus is on creating content in this genre.

    {ACTION}

    Regarding {SCENE}, please follow the steps below:

    1.Establish a background and main storyline that includes meaningful plot twists, significance, and detailed elements.
    2.Carefully examine the theme and identify any characters in the character library that conflict with it. If there are conflicts, create new characters according to the template.
    3.Based on the evaluation in step 2, determine which characters need to be newly created and which ones can be reused.
    4.Output the {LANG} background and {LANG} character settings, and translate the character settings into English.
    5.Species setting (in English only): if the character is human, specify as "boy" or "girl"; if it’s an animal, state the species name in English; for others, just use a single word like "item" — no extra text.
    6.Any tag containing # that appears before a colon (:) must be in English, to avoid issues with recognition in later stages.
    
    The character library is as follows. :
    {getRolesInfoGlobal()}

    The overall storyboard format should be as follows:
    # Background:
      ## background: Describe your idea in Chinese
    # Roles:
      ## role
        ###1 roleEng: English name
        ###1 roleName: A {LANG} name
        ###1 roleSpecies: English species name
        ###1 roleDesc: A simple appearance description in {LANG}
        ###1 roleTrans: English translation of the description
      ## role
        ###2 roleEng: English name
        ###2 roleName: A Chinese name
        ###2 roleSpecies: English species name
        ###2 roleDesc: A simple appearance description in Chinese
        ###2 roleTrans: English translation of the description
    # Shots:
      ## shotText:only {LANG} content,nothing else {LANG} tag
      ## shotText:only {LANG} content,nothing else {LANG} tag
      ...
    """
    return storyboard_prompt


def read_refine_shot_subtitle():
    refine_shot_subtitle = f"""
    Although this instructions are given in English, your thought process must be in {LANG}.

    Please act as the {SCENE} storyboard director.

    You need to refine the storyboard text for shot number {{0}} based on the outline provided by the main director:
    The character information in the outline includes:
    {{3}}

    {{2}}

    The text for the current shot is: {{1}}

    1. Distinguish from the previous text, only process the content for the current shot number; do not confuse it with the previous shot.
    2. remove camera shot directions description from current content.
    3. Connect with the previous text, output subtitle_text for subtitles and dubbing. Do not include any camera shot directions, and ensure the connecting words make it sound natural.
    4. subtitle_text should tell the story along the shot’s action line, avoid telling the story of multiple shots in one shot.
    5. Based on subtitle_text, imagine the first frame when the shot begins and provide a description in English.
    6. Based on subtitle_text's action line, there may be multiple connecting scenes, so you need to split the scenes.
    7. According to how many scenes need to be split, output the split_text from subtitle_text to help with time distribution for the scenes later.
    8. Based on split_text, provide scene_text that gives information to the video director, focusing on character action time, rather than literary style.
    9. Using scene_text, provide a video trend description (video_trends_prompt) to describe the development of the subsequent scenes, background, and actions.
    10. To prevent later issues with recognition, the characters in both scene_text and video_trends_prompt must be written using their English name or ID.

    Strictly follow the format below, no extra words:
    #subtitle_text:this is must be {LANG}
    ##first_frame_prompt:this is a prompt
    #Scenes:
      ## scene
        ##split_text:this must part of subtitle_text
        ##scene_text:this is must be {LANG}
        ##video_trends_prompt:this is a english prompt
      ## scene
        ##split_text:this must part of subtitle_text
        ##scene_text:this is must be {LANG}
        ##video_trends_prompt:this is a english prompt
    """
    return refine_shot_subtitle


if __name__ == '__main__':
    print(storyboard_prompt)