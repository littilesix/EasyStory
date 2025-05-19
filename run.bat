prompt (easyStory) $P$G

set workPath=%~dp0

:: Automatically switch to the correct drive
cd /d %workPath%

:: Update PATH 
set PATH=%workPath%/env;%workPath%/env/Scripts;%PATH%

python modules/StoryTask.py

cmd