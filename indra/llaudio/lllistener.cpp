/**
 * @file listener.cpp
 * @brief Implementation of LISTENER class abstracting the audio support
 *
 * $LicenseInfo:firstyear=2000&license=viewerlgpl$
 * Second Life Viewer Source Code
 * Copyright (C) 2010, Linden Research, Inc.
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

#include "linden_common.h"

#include "lllistener.h"

#define DEFAULT_AT  0.0f,0.0f,-1.0f
#define DEFAULT_UP  0.0f,1.0f,0.0f

//-----------------------------------------------------------------------
// constructor
//-----------------------------------------------------------------------
LLListener::LLListener()
    : mPosition(LLVector3::zero),
      mVelocity(LLVector3::zero),
      mListenAt(DEFAULT_AT),
      mListenUp(DEFAULT_UP)
{
}

//-----------------------------------------------------------------------
void LLListener::translate(const LLVector3& offset)
{
    mPosition += offset;
}

//-----------------------------------------------------------------------
void LLListener::setPosition(const LLVector3& pos)
{
    mPosition = pos;
}

//-----------------------------------------------------------------------
LLVector3 LLListener::getPosition(void)
{
    return(mPosition);
}

//-----------------------------------------------------------------------
LLVector3 LLListener::getAt(void)
{
    return(mListenAt);
}

//-----------------------------------------------------------------------
LLVector3 LLListener::getUp(void)
{
    return(mListenUp);
}

//-----------------------------------------------------------------------
void LLListener::setVelocity(const LLVector3& vel)
{
    mVelocity = vel;
}

//-----------------------------------------------------------------------
void LLListener::orient(const LLVector3& up, const LLVector3& at)
{
    mListenUp = up;
    mListenAt = at;
}

//-----------------------------------------------------------------------
void LLListener::set(const LLVector3& pos, const LLVector3& vel, const LLVector3& up, const LLVector3& at)
{
    setPosition(pos);
    setVelocity(vel);
    orient(up,at);
}

//-----------------------------------------------------------------------
void LLListener::setDopplerFactor(F32 factor)
{
}

//-----------------------------------------------------------------------
F32 LLListener::getDopplerFactor()
{
    return (1.f);
}

//-----------------------------------------------------------------------
void LLListener::setRolloffFactor(F32 factor)
{
}

//-----------------------------------------------------------------------
F32 LLListener::getRolloffFactor()
{
    return (1.f);
}

//-----------------------------------------------------------------------
void LLListener::commitDeferredChanges()
{
}

