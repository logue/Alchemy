/*
 * @file alpanelaomini.h
 * @brief Animation overrides mini control panel
 *
 * Copyright (c) 2016, Cinder Roxley <cinder@sdf.org>
 * Copyright (C) 2020, Rye Mutt <rye@alchemyviewer.org>
 *
 * Permission is hereby granted, free of charge, to any person or organization
 * obtaining a copy of the software and accompanying documentation covered by
 * this license (the "Software") to use, reproduce, display, distribute,
 * execute, and transmit the Software, and to prepare derivative works of the
 * Software, and to permit third-parties to whom the Software is furnished to
 * do so, all subject to the following:
 *
 * The copyright notices in the Software and this entire statement, including
 * the above license grant, this restriction and the following disclaimer,
 * must be included in all copies of the Software, in whole or in part, and
 * all derivative works of the Software, unless such copies or derivative
 * works are solely in the form of machine-executable object code generated by
 * a source language processor.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 */

#ifndef AL_PANELAOMINI_H
#define AL_PANELAOMINI_H

#include "llpanel.h"

class LLComboBox;
//class LLUICtrl;

class ALPanelAOMini final : public LLPanel
{
public:
    ALPanelAOMini();

    BOOL postBuild() override;

protected:
    ~ALPanelAOMini();

private:
    void onSelectSet(const LLSD& userdata);
    void onClickSit(const LLSD& userdata);
    void onClickNext();
    void onClickPrevious();
    void openAOFloater();
    void updateSetList();
    void onSetChanged(const std::string& set_name);

    LLComboBox* mSetList;

    boost::signals2::connection mReloadCallback;
    boost::signals2::connection mSetChangedCallback;
};

#endif // LL_PANELAOMINI_H
