module;

#include "primitive_types.hpp"

export module vulkan_app:extra;

import vulkan_hpp;
import std;

struct TextArea {
  u32 width;           // px
  u32 height;          // px
  u32 glyphWidth{48};  // px
  u32 glyphHeight{48}; // px
  u32 numberOfColumns{width / glyphWidth};
  u32 numberOfRows{height / glyphHeight};
  u32 logicalLineWidthLimit = 120; // number of glyphs
  bool uiActiveArea = false;

  std::vector<char *> buffer;     // actual text
  std::vector<char *> lineSplits; // each \n for fast traversal between lines
  char *visibleAreaBegin;         // first visible row
  char *visibleAreaEnd;           // next row after last visible row

  void setNewDimensions(u32 newWidth, u32 newHeight) {
    if (newWidth < 0 || newHeight < 0) {
      return;
    }

    width = newWidth;
    height = newHeight;

    numberOfColumns = width / glyphWidth;
    numberOfRows = height / glyphHeight;
  }

  u32 getLogicalLineWidthBegin(char *lineBegin) {
    auto res = std::find(lineSplits.begin(), lineSplits.end(), lineBegin);
    return (res + 1) - res;
  }

  u32 getLogicalLineWidthEnd(char *lineEnd) {
    auto res = std::find(lineSplits.begin(), lineSplits.end(), lineEnd);
    return res - (res - 1);
  }

  u32 getLogicalLineWidth(char *lineSplit) { return (lineSplit + 1) - lineSplit; }

  void recalculateLogicalLineWidths(u32 newLogicalLineWidthLimit) {
    auto it = buffer.begin();
    while (it != buffer.end()) {
    }
  };
  void recalculateVisibleArea();
};
