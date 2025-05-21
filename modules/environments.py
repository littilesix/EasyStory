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

def read_storyboard_prompt(style: str):
    storyboard_prompt = f"""
    Although these instructions are written in English, your thought process and output must be in {LANG}.

    You are a chief director specializing in videos related to {SCENE}, with a focus on creating content in the {style} genre.

    {ACTION}

    For the topic of {SCENE}, follow the steps below:

    1. Establish a background and main storyline that includes meaningful plot twists, thematic depth, and rich details.
    2. Carefully examine the genre "{style}" and identify any roles in the role library whose styles conflict with it. If there are conflicts, create new roles following the format.
    3. Based on step 2, decide which roles need to be newly created and which can be reused.
    4. Output the background and character descriptions in {LANG}, and also provide English translations of the character descriptions.
    5. Species setting (in English only): if the character is human, specify "boy" or "girl"; if it is an animal, provide the species name in English; for other cases, use a single English noun like "item" — no additional explanation is needed.
    6. Any tag that appears **before a colon (:)** and contains a "#" must be in English to ensure compatibility with later recognition stages.

    The current role library is as follows:
    {getRolesInfoGlobal()}

    Please use the following storyboard format exactly:

    # Background:
      ## background: Describe your idea in {LANG}

    # Roles:
      ## role
        ###1 roleEng: English name
        ###1 roleName: Name in {LANG}
        ###1 roleSpecies: Species in English
        ###1 roleDesc: Simple appearance description in {LANG}
        ###1 roleTrans: English translation of the description
      ## role
        ###2 roleEng: English name
        ###2 roleName: Name in {LANG}
        ###2 roleSpecies: Species in English
        ###2 roleDesc: Simple appearance description in {LANG}
        ###2 roleTrans: English translation of the description

    # Shots:
      ## shotText: Only content in {LANG}. No other language or tags allowed.
      ## shotText: Only content in {LANG}. No other language or tags allowed.
      ...
    """
    return storyboard_prompt

def read_refine_shot_subtitle():
    refine_shot_subtitle = f"""
    Although these instructions are given in English, your thought process must be in {LANG}.

    Please act as the {SCENE} storyboard director.

    You need to refine the storyboard text for shot number {{0}} based on the outline provided by the main director:
    The character information in the outline includes:
    {{3}}

    {{2}}

    The text for the current shot is:
    {{1}}

    Please automatically determine the appropriate type of continuity (emotional flow, action continuity, chronological flow, or scene transition) based on the previous subtitle_text and current shot text.

    You must:
    1. Distinguish from the previous text; only process the content for the current shot number.
    2. Remove any camera shot direction descriptions from the current content.
    3. Ensure the subtitle_text logically continues the story from the previous subtitle_text.
    4. Add a natural transitional phrase at the beginning if it improves the flow.
    5. Use appropriate transitional phrases based on the determined continuity type. You may refer to these examples:
       - Emotional flow: "Still shaken,", "With a heavy heart,", "Tears still in her eyes,"
       - Action continuity: "He quickly turns around,", "Without hesitation,", "In the next instant,"
       - Chronological flow: "Moments later,", "At the same time,", "Later that night,"
       - Scene transition: "Meanwhile,", "Elsewhere,", "On the other side of town,"

    6. Output `subtitle_text` for subtitles and dubbing in {LANG}. Do not include camera shot directions.
    7. `subtitle_text` should only tell the story of the current shot’s action line, not multiple shots.
    8. Based on subtitle_text, imagine the first frame when the shot begins and provide a description in English.
    9. Based on the subtitle_text's action line, split into multiple scenes if needed.
    10. According to the split scenes, output `split_text` to help with time distribution later.
    11. Based on `split_text`, write `scene_text` in {LANG} that gives action-timing instructions to the video director.
    12. Using `scene_text`, provide a `video_trends_prompt` to describe the development of each scene in English.
    13. Characters in both `scene_text` and `video_trends_prompt` must be referred to using their English names or IDs to avoid ambiguity.

    Strictly follow the format below, no extra words:
    #subtitle_text:this is must be {LANG}
    ##first_frame_prompt:this is a prompt
    #Scenes:
      ## scene
        ##split_text:this must part of subtitle_text
        ##scene_text:this is must be {LANG}
        ##video_trends_prompt:this is an English prompt
      ## scene
        ##split_text:this must part of subtitle_text
        ##scene_text:this is must be {LANG}
        ##video_trends_prompt:this is an English prompt
    """
    return refine_shot_subtitle

"""
def read_refine_shot_subtitle():
    refine_shot_subtitle = f"
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
    "
    return refine_shot_subtitle
"""

if __name__ == '__main__':
    print(storyboard_prompt)