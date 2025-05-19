import gc, torch

def cleanMsg(obj,*attrs):
    print(f"{obj.__class__}'s members:{','.join(attrs)}.Memory cleaned up.")

def clean(obj,attr:str):
    if hasattr(obj, attr):
        delattr(obj, attr)
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cleanMsg(obj,attr)

def cleanAll(obj, *attrs):
    
    for attr in attrs:
        if hasattr(obj, attr):
            delattr(obj, attr)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cleanMsg(obj,*attrs)

def testImageGen(prompt,lora = None,weight=1,count=5):
    from TextGen import TextGener
    text = TextGener()
    prompt = text.prompt(prompt)
    text.close()
    from ImageGen import ImageGener
    imgPipeline = ImageGener()
    imgPipeline.weight = weight
    if lora:
        imgPipeline.load_lora(lora)
    for i in range(count):
        image = imgPipeline(prompt=prompt,width=600,height=800)
        image.save(f"{i}.png")
    #image.show()
    return imgPipeline,prompt

def testVideoGen(image:str,prompt:str):
    from VideoGen import VideoGener
    video = VideoGener()
    video.run(image,prompt)