/**
 * @file llteleporthistorystorage.h
 * @brief Saving/restoring of teleport history.
 *
 * $LicenseInfo:firstyear=2009&license=viewerlgpl$
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

#ifndef LL_LLTELEPORTHISTORYSTORAGE_H
#define LL_LLTELEPORTHISTORYSTORAGE_H

#include <vector>

#include "llsingleton.h"
#include "lldate.h"
#include "v3dmath.h"

class LLSD;

/**
 * An item of the teleport history, stored in file
 *
 * Contains the location's global coordinates, title and date.
 */
class LLTeleportHistoryPersistentItem
{
public:
    LLTeleportHistoryPersistentItem() = default;

    LLTeleportHistoryPersistentItem(std::string grid, std::string region, std::string title, const LLVector3& local_pos, const LLVector3d& global_pos)
        : mGrid(std::move(grid)), mRegion(std::move(region)), mTitle(std::move(title)), mGlobalPos(global_pos), mLocalPos(local_pos), mDate(LLDate::now())
    {}

    LLTeleportHistoryPersistentItem(std::string grid, std::string region, std::string title, const LLVector3& local_pos, const LLVector3d& global_pos, const LLDate& date)
        : mGrid(std::move(grid)), mRegion(std::move(region)), mTitle(std::move(title)), mGlobalPos(global_pos), mLocalPos(local_pos), mDate(date)
    {}

    LLTeleportHistoryPersistentItem(const LLSD& val);
    LLSD toLLSD() const;

    std::string mGrid;
    std::string mRegion;
    std::string mTitle;
    LLVector3d  mGlobalPos;
    LLVector3   mLocalPos;
    LLDate      mDate;
};

/**
 * Persistent teleport history.
 *
 */
class LLTeleportHistoryStorage final : public LLSingleton<LLTeleportHistoryStorage>
{
    LLSINGLETON(LLTeleportHistoryStorage);
    ~LLTeleportHistoryStorage();
    LOG_CLASS(LLTeleportHistoryStorage);

public:

    typedef std::vector<LLTeleportHistoryPersistentItem> slurl_list_t;

    // removed_index is index of removed item, which replaced by more recent
    typedef boost::function<void(S32 removed_index)>        history_callback_t;
    typedef boost::signals2::signal<void(S32 removed_index)>    history_signal_t;

    /**
     * @return history items.
     */
    const slurl_list_t& getItems() const { return mItems; }
    void            purgeItems();

    void addItem(const std::string grid, const std::string region, const std::string title, const LLVector3& local_pos, const LLVector3d& global_pos);
    void addItem(const std::string grid, const std::string region, const std::string title, const LLVector3& local_pos, const LLVector3d& global_pos, const LLDate& date);

    void removeItem(S32 idx);

    void save();

    /**
     * Set a callback to be called upon history changes.
     *
     * Multiple callbacks can be set.
     */
    boost::signals2::connection setHistoryChangedCallback(history_callback_t cb);

    /**
     * Go to specific item in the history.
     *
     * The item is specified by its index (starting from 0).
     */
    void                    goToItem(S32 idx);

    /**
     * Show specific item on map.
     *
     * The item is specified by its index (starting from 0).
     */
    void                    showItemOnMap(S32 idx);

//private:
// [RLVa:KB] - Checked: 2010-09-03 (RLVa-1.2.1b) | Added: RLVa-1.2.1b
protected:
    friend class RlvUIEnabler;
// [/RLVa:KB]

    void load();
    void dump() const;

    void onTeleportHistoryChange();
    bool compareByTitleAndGlobalPos(const LLTeleportHistoryPersistentItem& a, const LLTeleportHistoryPersistentItem& b);
    void onRegionNameResponse(
        std::string region_name,
        LLVector3 local_coords,
        U64 region_handle, const std::string& url, const LLUUID& snapshot_id, bool teleport);

    slurl_list_t    mItems;
    std::string mFilename;

    /**
     * Signal emitted when the history gets changed.
     *
     * Invokes callbacks set with setHistoryChangedCallback().
     */
    history_signal_t        mHistoryChangedSignal;
};

#endif //LL_LLTELEPORTHISTORYSTORAGE_H
