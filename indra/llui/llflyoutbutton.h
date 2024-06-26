/**
 * @file llflyoutbutton.h
 * @brief LLFlyoutButton base class
 *
 * $LicenseInfo:firstyear=2001&license=viewerlgpl$
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

// A control that displays the name of the chosen item, which when clicked
// shows a scrolling box of choices.

#ifndef LL_LLFLYOUTBUTTON_H
#define LL_LLFLYOUTBUTTON_H

#include "llcombobox.h"

// Classes

class LLFlyoutButton : public LLComboBox
{
public:
    struct Params : public LLInitParam::Block<Params, LLComboBox::Params>
    {
        Optional<LLButton::Params> action_button;
// [SL:KB] - Patch: Control-FlyoutButton | Checked: 2012-08-09 (Catznip-3.3)
        Optional<S32>              arrow_button_width;
// [/SL:KB]
        Deprecated                 allow_text_entry;

        Params()
        :   action_button("action_button"),
// [SL:KB] - Patch: Control-FlyoutButton | Checked: 2012-08-09 (Catznip-3.3)
            arrow_button_width("arrow_button_width", 24),
// [/SL:KB]
            allow_text_entry("allow_text_entry")
        {
            changeDefault(LLComboBox::Params::allow_text_entry, false);
        }

    };
protected:
    LLFlyoutButton(const Params&);
    friend class LLUICtrlFactory;
public:
    virtual void    draw() override;
// [SL:KB] - Patch: Control-FlyoutButton | Checked: Catznip-6.4
            void    setLabel(const LLStringExplicit& name) override;
            void    updateLabel() override;
// [/SL:KB]

    void setToggleState(BOOL state);

    void onActionButtonClick(const LLSD& data);

protected:
    LLButton*               mActionButton;
    BOOL                    mToggleState;
};

#endif // LL_LLFLYOUTBUTTON_H
