# -*- cmake -*-

include(Variables)
include(GLEXT)
include(Prebuilt)
include(SDL3)

include_guard()

if (LINUX)
  # linux uses SDL3 for window and keyboard
  target_compile_definitions( ll::SDL3 INTERFACE LL_USE_SDL_WINDOW=1 LL_USE_SDL_KEYBOARD=1 )
endif (LINUX)

