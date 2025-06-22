module;

#include "macros.hpp"
#include "primitive_types.hpp"

export module vulkan_app:TextArea;

import vulkan_hpp;
import std;

import :PieceTable;

/**
 * @class TextArea
 * @brief Represents the logical state of a text area, now powered by a PieceTable.
 *
 * This class manages the text content via a PieceTable for efficient editing,
 * and handles layout, scrolling, and determining the visible portion of the text.
 */
class TextArea {
public:
  TextArea(u32 width, u32 height, u32 glyphWidth = 10, u32 glyphHeight = 20)
      : m_width(width), m_height(height), m_glyphWidth(glyphWidth), m_glyphHeight(glyphHeight),
        m_text_buffer("") // Initialize with empty content
  {
    updateDimensions();
  }

  // --- Public Interface ---

  void setPixelDimensions(u32 newWidth, u32 newHeight) {
    if (newWidth > 0 && newHeight > 0) {
      m_width = newWidth;
      m_height = newHeight;
      updateDimensions();
    }
  }

  // Modification functions now operate on the PieceTable
  void setText(std::string text) {
    m_text_buffer = PieceTable(std::move(text));
    recalculateLineSplits();
  }

  void insertText(size_t pos, std::string_view text) {
    m_text_buffer.insert(pos, text);
    recalculateLineSplits(); // In a real editor, this would be more optimized
  }

  void deleteText(size_t pos, size_t length) {
    m_text_buffer.remove(pos, length);
    recalculateLineSplits(); // In a real editor, this would be more optimized
  }

  void scroll(int lineDelta) {
    if (m_lines.empty()) {
      m_firstVisibleLine = 0;
      return;
    }

    long long new_line = static_cast<long long>(m_firstVisibleLine) + lineDelta;

    // Clamp the value
    if (new_line < 0) {
      new_line = 0;
    }
    if (new_line >= m_lines.size()) {
      new_line = m_lines.size() - 1;
    }
    m_firstVisibleLine = static_cast<size_t>(new_line);
  }

  // --- Getters ---

  /**
   * @return The lines of text currently visible in the text area.
   * @note This reconstructs the visible lines from the PieceTable.
   */
  std::vector<std::string> getVisibleLines() const {
    if (m_lines.empty()) {
      return {};
    }

    std::vector<std::string> visible_lines;
    visible_lines.reserve(m_numberOfRows);

    auto start_it = m_lines.begin() + m_firstVisibleLine;
    auto end_it = std::min(start_it + m_numberOfRows, m_lines.end());

    for (auto it = start_it; it != end_it; ++it) {
      visible_lines.push_back(*it);
    }

    return visible_lines;
  }

  size_t getTotalLineCount() const { return m_lines.size(); }
  u32 getColumns() const { return m_numberOfColumns; }
  u32 getRows() const { return m_numberOfRows; }

  /**
   * @return The entire text content as a single string.
   * @note Potentially slow. Best for saving, not for display updates.
   */
  std::string getText() const { return m_text_buffer.to_string(); }

private:
  void updateDimensions() {
    if (m_glyphWidth > 0)
      m_numberOfColumns = m_width / m_glyphWidth;
    if (m_glyphHeight > 0)
      m_numberOfRows = m_height / m_glyphHeight;
    scroll(0); // Re-validate scroll position
  }

  /**
   * @brief Re-parses the document from the PieceTable to find line breaks.
   * @note This is a naive implementation. A production-quality editor would
   * integrate line management more deeply with the text buffer.
   */
  void recalculateLineSplits() {
    m_lines.clear();
    // Reconstruct the full string to split it. This is the simplest,
    // but not the most performant way. A better way would be to
    // iterate through pieces and find newlines.
    std::string full_text = m_text_buffer.to_string();
    if (full_text.empty()) {
      return;
    }

    std::string_view view(full_text);
    size_t start = 0;
    size_t pos;

    while ((pos = view.find('\n', start)) != std::string_view::npos) {
      m_lines.push_back(std::string(view.substr(start, pos - start)));
      start = pos + 1;
    }

    if (start < view.length()) {
      m_lines.push_back(std::string(view.substr(start)));
    }

    // Ensure the scroll position is still valid.
    scroll(0);
  }

  // Physical Properties
  u32 m_width;
  u32 m_height;
  u32 m_glyphWidth;
  u32 m_glyphHeight;

  // Logical Properties
  u32 m_numberOfColumns = 0;
  u32 m_numberOfRows = 0;

  // Text Data & State
  PieceTable m_text_buffer;         // The powerful text buffer
  std::vector<std::string> m_lines; // Cached lines for display
  size_t m_firstVisibleLine = 0;    // Index of the first visible line in m_lines
};
