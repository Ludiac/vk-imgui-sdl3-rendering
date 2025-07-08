module;

#include "macros.hpp"
#include "primitive_types.hpp"
#include <glm/glm.hpp>

export module vulkan_app:TextView;

import std;
import :TextArea; // The module for TextEditor
import :text;     // The module for Font

// Represents the visual style for a range of text.
// For now, it only contains color, but can be expanded with font style (bold, italic), etc.
export struct TextStyle {
    glm::vec4 color{1.0f, 1.0f, 1.0f, 1.0f}; // Default to white
    // bool is_bold = false;
    // bool is_italic = false;
};

// Defines a styled range of text using character positions.
export struct StyledRange {
    size_t start;
    size_t length;
    TextStyle style;

    // For sorting and searching ranges
    bool operator<(const StyledRange& other) const {
        return start < other.start;
    }
};

/**
 * @class TextView
 * @brief Manages the visible state and styling of a text document.
 *
 * Acts as the "View" in an MVC pattern. It does not own the text data itself
 * (that's the TextEditor's job), but it's responsible for:
 * 1.  Managing the scroll position to determine which lines are visible.
 * 2.  Storing and applying style information (like syntax highlighting) to ranges of text.
 * 3.  Providing the list of visible lines and their styles to a rendering system.
 */
export class TextView {
public:
    /**
     * @brief Constructs a TextView.
     * @param editor A reference to the TextEditor model.
     * @param font A reference to the Font used for metrics.
     */
    TextView(TextEditor& editor, Font& font)
        : m_editor(editor), m_font(font) {}

    // --- Configuration ---

    /**
     * @brief Sets the dimensions of the view area in pixels.
     */
    void setDimensions(float width, float height) {
        m_width = width;
        m_height = height;
        // In a real implementation, we might recalculate visible lines here.
    }

    // --- Scrolling ---

    /**
     * @brief Scrolls the view vertically by a number of lines.
     * @param delta_lines Positive to scroll down, negative to scroll up.
     */
    void scroll(i32 delta_lines) {
        if (delta_lines > 0) {
            // Scroll down, but don't go past the end of the document
            size_t max_first_line = m_editor.lineCount() > 0 ? m_editor.lineCount() - 1 : 0;
            m_first_visible_line = std::min(m_first_visible_line + (size_t)delta_lines, max_first_line);
        } else if (delta_lines < 0) {
            // Scroll up, but don't go below the start
            size_t scroll_amount = -delta_lines;
            m_first_visible_line = (m_first_visible_line > scroll_amount) ? m_first_visible_line - scroll_amount : 0;
        }
    }

    // --- Styling ---

    /**
     * @brief Applies a style to a specified range of characters.
     * @note This is a simple implementation. A real editor would need to handle
     *       merging and splitting overlapping style ranges.
     */
    void applyStyle(size_t start, size_t length, const TextStyle& style) {
        // For simplicity, we just add a new range.
        // A more robust implementation would merge this with existing ranges.
        m_styles.emplace_back(StyledRange{start, length, style});
        // Keep styles sorted for efficient lookup.
        std::sort(m_styles.begin(), m_styles.end());
    }

    /**
     * @brief Clears all custom styling.
     */
    void clearStyles() {
        m_styles.clear();
    }

    /**
     * @brief Gets the style for a character at a specific position.
     * @return The applied style, or a default style if none is found.
     */
    TextStyle getStyleAt(size_t char_pos) const {
        // Find the last style that starts at or before char_pos
        auto it = std::upper_bound(m_styles.begin(), m_styles.end(), StyledRange{char_pos, 0, {}});
        if (it != m_styles.begin()) {
            --it; // Move to the potential containing range
            // Check if the position is actually within this range
            if (char_pos < it->start + it->length) {
                return it->style;
            }
        }
        return m_default_style; // Return default if no style applies
    }

    // --- Data Retrieval for Rendering ---

    /**
     * @return The starting line index of the visible area.
     */
    size_t getFirstVisibleLine() const {
        return m_first_visible_line;
    }

    /**
     * @return The number of lines that can fit in the view's current height.
     */
    size_t getVisibleLineCount() const {
        if (m_font.atlasData.lineHeight <= 0) {
            return 20; // Fallback
        }
        // This is a simplified calculation. A real implementation would use font metrics.
        const double pointSize = 36.0;
        const double fontUnitToPixelScale = pointSize * (96.0 / 72.0 * 2) / m_font.atlasData.unitsPerEm;
        const double line_height_px = m_font.atlasData.lineHeight * fontUnitToPixelScale;

        if (line_height_px <= 0) {
            return 20; // Fallback
        }
        return static_cast<size_t>(m_height / line_height_px) + 1;
    }

private:
    TextEditor& m_editor;
    Font& m_font;

    // Viewport state
    float m_width = 0.0f;
    float m_height = 0.0f;
    size_t m_first_visible_line = 0;
    float m_horizontal_scroll_offset_px = 0.0f;

    // Styling information
    TextStyle m_default_style;
    std::vector<StyledRange> m_styles;
};
