/** 
 * @file llscrolllistcell.h
 * @brief Scroll lists are composed of rows (items), each of which 
 * contains columns (cells).
 *
 * $LicenseInfo:firstyear=2007&license=viewerlgpl$
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

#ifndef LLSCROLLLISTCELL_H
#define LLSCROLLLISTCELL_H

#include "llfontgl.h"		// HAlign
#include "llpointer.h"		// LLPointer<>
#include "lluistring.h"
#include "v4color.h"
#include "llui.h"

#include "lllineeditor.h"

class LLCheckBoxCtrl;
class LLSD;
class LLUIImage;

/*
 * Represents a cell in a scrollable table.
 *
 * Sub-classes must return height and other properties 
 * though width accessors are implemented by the base class.
 * It is therefore important for sub-class constructors to call
 * setWidth() with realistic values.
 */
class LLScrollListCell
{
public:
	struct Params : public LLInitParam::Block<Params>
	{
		Optional<std::string>		type,
									column;

		Optional<S32>				width;
		Optional<bool>				enabled,
									visible;

		Optional<void*>				userdata;
		Optional<LLSD>				value; // state of checkbox, icon id/name, date
		Optional<LLSD>				alt_value;
		Optional<std::string>		label; // description or text
		Optional<std::string>		tool_tip;

		Optional<const LLFontGL*>	font;
		Optional<LLColor4>			font_color;
		Optional<LLFontGL::HAlign>	font_halign;

		Optional<LLColor4>			color;

		Params()
		:	type("type", "text"),
			column("column"),
			width("width"),
			enabled("enabled", true),
			visible("visible", true),
			value("value"),
			alt_value("alt_value", ""),
			label("label"),
			tool_tip("tool_tip", ""),
			font("font", LLFontGL::getFontSansSerifSmall()),
			font_color("font_color", LLColor4::black),
			color("color", LLColor4::white),
			font_halign("halign", LLFontGL::LEFT)
		{
			addSynonym(column, "name");
			addSynonym(font_color, "font-color");
		}
	};

	static LLScrollListCell* create(const Params&);

	LLScrollListCell(const LLScrollListCell::Params&);
	virtual ~LLScrollListCell() = default;

	virtual void			draw(const LLColor4& color, const LLColor4& highlight_color) const {};		// truncate to given width, if possible
	virtual S32				getWidth() const {return mWidth;}
	virtual S32				getContentWidth() const { return 0; }
	virtual S32				getHeight() const { return 0; }
	virtual const LLSD		getValue() const;
	virtual const LLSD		getAltValue() const;
	virtual void			setValue(const LLSD& value) { }
	virtual void			setAltValue(const LLSD& value) { }
	virtual const std::string &getToolTip() const { return mToolTip; }
	virtual void			setToolTip(const std::string &str) { mToolTip = str; }
	virtual BOOL			getVisible() const { return TRUE; }
	virtual void			setWidth(S32 width) { mWidth = width; }
	virtual void			highlightText(S32 offset, S32 num_chars) {}
	virtual BOOL			isText() const { return FALSE; }
	virtual BOOL			needsToolTip() const { return ! mToolTip.empty(); }
	virtual void			setColor(const LLColor4&) {}
	virtual void			onCommit() {};

	virtual BOOL			handleClick() { return FALSE; }
	virtual	void			setEnabled(BOOL enable) { }

private:
	S32 mWidth;
	std::string mToolTip;
};

class LLScrollListSpacer : public LLScrollListCell
{
public:
	LLScrollListSpacer(const LLScrollListCell::Params& p) : LLScrollListCell(p) {}
	/*virtual*/ ~LLScrollListSpacer() = default;
	/*virtual*/ void			draw(const LLColor4& color, const LLColor4& highlight_color) const {}
};

/*
 * Cell displaying a text label.
 */
class LLScrollListText : public LLScrollListCell
{
public:
	LLScrollListText(const LLScrollListCell::Params&);
	/*virtual*/ ~LLScrollListText();

	/*virtual*/ void    draw(const LLColor4& color, const LLColor4& highlight_color) const;
	/*virtual*/ S32		getContentWidth() const;
	/*virtual*/ S32		getHeight() const;
	/*virtual*/ void	setValue(const LLSD& value);
	/*virtual*/ void	setAltValue(const LLSD& value);
	/*virtual*/ const LLSD getValue() const;
	/*virtual*/ const LLSD getAltValue() const;
	/*virtual*/ BOOL	getVisible() const;
	/*virtual*/ void	highlightText(S32 offset, S32 num_chars);

	/*virtual*/ void	setColor(const LLColor4&);
	/*virtual*/ BOOL	isText() const;
	/*virtual*/ const std::string &	getToolTip() const;
	/*virtual*/ BOOL	needsToolTip() const;

	S32				getTextWidth() const { return mTextWidth;}
	void			setTextWidth(S32 value) { mTextWidth = value;} 
	virtual void	setWidth(S32 width) { LLScrollListCell::setWidth(width); mTextWidth = width; }

	void			setText(const LLStringExplicit& text);
	void			setFontStyle(const U8 font_style);

protected:
	LLUIString		mText;
	LLUIString		mAltText;
	S32				mTextWidth;
	const LLFontGL*	mFont;
	LLColor4		mColor;
	LLColor4		mHighlightColor;
	U8				mUseColor;
	LLFontGL::HAlign mFontAlignment;
	BOOL			mVisible;
	S32				mHighlightCount;
	S32				mHighlightOffset;

	LLPointer<LLUIImage> mRoundedRectImage;

	static U32 sCount;
};

/*
 * Cell displaying an image. AT the moment, this is specifically UI image
 */
class LLScrollListIcon : public LLScrollListCell
{
public:
	LLScrollListIcon(const LLScrollListCell::Params& p);
	/*virtual*/ ~LLScrollListIcon() = default;
	/*virtual*/ void	draw(const LLColor4& color, const LLColor4& highlight_color) const override;
	/*virtual*/ S32		getWidth() const override;
	/*virtual*/ S32		getHeight() const override;
	/*virtual*/ const LLSD		getValue() const override;
	/*virtual*/ void	setColor(const LLColor4&) override;
	/*virtual*/ void	setValue(const LLSD& value) override;

	void setClickCallback(BOOL (*callback)(void*), void* user_data);
	BOOL handleClick() override;

private:
	LLPointer<LLUIImage>	mIcon;
	LLColor4				mColor;
	LLFontGL::HAlign		mAlignment;

	BOOL (*mCallback)(void*);
	void* mUserData;
};

/*
 * An interactive cell containing a check box.
 */
class LLScrollListCheck : public LLScrollListCell
{
public:
	LLScrollListCheck( const LLScrollListCell::Params&);
	/*virtual*/ ~LLScrollListCheck();
	/*virtual*/ void	draw(const LLColor4& color, const LLColor4& highlight_color) const;
	/*virtual*/ S32		getHeight() const			{ return 0; } 
	/*virtual*/ const LLSD	getValue() const;
	/*virtual*/ void	setValue(const LLSD& value);
	/*virtual*/ void	onCommit();

	/*virtual*/ BOOL	handleClick();
	/*virtual*/ void	setEnabled(BOOL enable);

	LLCheckBoxCtrl*	getCheckBox()				{ return mCheckBox; }

private:
	LLCheckBoxCtrl* mCheckBox;
};

class LLScrollListDate : public LLScrollListText
{
public:
	LLScrollListDate( const LLScrollListCell::Params& p );
	virtual void	setValue(const LLSD& value);
	virtual const LLSD getValue() const;

private:
	LLDate		mDate;
};

/*
* Cell displaying icon and text.
*/

class LLScrollListIconText : public LLScrollListText
{
public:
    LLScrollListIconText(const LLScrollListCell::Params& p);
    /*virtual*/ ~LLScrollListIconText() = default;
    /*virtual*/ void	draw(const LLColor4& color, const LLColor4& highlight_color) const;
    /*virtual*/ const LLSD		getValue() const;
    /*virtual*/ void	setValue(const LLSD& value);


    S32					getIconWidth() const;
    /*virtual*/ void	setWidth(S32 width);/* { LLScrollListCell::setWidth(width); mTextWidth = width - ; }*/

private:
    LLPointer<LLUIImage>	mIcon;
    S32						mPad;
};

class LLScrollListLineEditor : public LLScrollListCell
{
public:
	LLScrollListLineEditor( const LLScrollListCell::Params&);
	/*virtual*/ ~LLScrollListLineEditor();
	void	draw(const LLColor4& color, const LLColor4& highlight_color) const override;
	S32		getHeight() const override { return 0; }
	const LLSD	getValue() const override { return mLineEditor->getValue(); }
	void	setValue(const LLSD& value) override { mLineEditor->setValue(value); }
	void	onCommit() override { mLineEditor->onCommit(); }
	BOOL	handleClick() override;
	virtual BOOL	handleUnicodeChar(llwchar uni_char, BOOL called_from_parent);
	virtual BOOL	handleUnicodeCharHere(llwchar uni_char );
	void	setEnabled(BOOL enable) override { mLineEditor->setEnabled(enable); }

	LLLineEditor*	getLineEditor()				{ return mLineEditor; }
	BOOL	isText() const override { return FALSE; }

private:
	LLLineEditor* mLineEditor;
};

#endif
