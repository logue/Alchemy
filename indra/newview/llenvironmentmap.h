/**
 * @file llenvironmentmap.h
 * @brief LLEnvironmentMap class declaration
 *
 * $LicenseInfo:firstyear=2022&license=viewerlgpl$
 * Second Life Viewer Source Code
 * Copyright (C) 2022, Linden Research, Inc.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation;
 * version 2.1 of the License only.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Linden Research, Inc., 945 Battery Street, San Francisco, CA  94111  USA
 * $/LicenseInfo$
 */

#pragma once

#include "llcubemap.h"

class LLEnvironmentMap
{
public:
    // allocate an environment map of the given resolution 
    LLEnvironmentMap();

    // update this environment map
    // origin - position in agent space to generate environment map from in agent space
    // resolution - size of cube map to generate
    void update(const LLVector3& origin, U32 resolution);
    
    // cube map used to sample this environment map
    LLPointer<LLCubeMap> mCubeMap;

    // point at which environment map was generated from (in agent space)
    LLVector3 mOrigin;
};

