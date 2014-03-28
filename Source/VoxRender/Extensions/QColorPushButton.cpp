
#include "QColorPushButton.h"

#include <QPainter>
#include <QtWidgets/QColorDialog>
#include <QPaintEvent>

QColorPushButton::QColorPushButton(QWidget* pParent) :
    QPushButton(pParent),
    m_margin(7),
    m_radius(4),
    m_color(Qt::white)
{
    setText("");
}

QSize QColorPushButton::sizeHint() const
{
    return QSize(50, 30);
}

void QColorPushButton::paintEvent(QPaintEvent* pPaintEvent)
{
    setText("");

    QPushButton::paintEvent(pPaintEvent);

    QPainter painter(this);

    // Get button rectangle
    QRect ColorRectangle = pPaintEvent->rect();

    // Deflate it
    ColorRectangle.adjust(m_margin, m_margin, -m_margin, -m_margin);

    // Use anti aliasing
    painter.setRenderHint(QPainter::Antialiasing);

    // Rectangle styling
    painter.setBrush(QBrush(isEnabled() ? m_color : Qt::lightGray));
    painter.setPen(QPen(isEnabled() ? QColor(25, 25, 25) : Qt::darkGray, 0.5));

    // Draw
    painter.drawRoundedRect(ColorRectangle, m_radius, Qt::AbsoluteSize);
}

void QColorPushButton::mousePressEvent(QMouseEvent* pEvent)
{
    static int check = 0;
    check++;

    QColorDialog ColorDialog;

    auto originalColor = m_color;

    emit beginColorSelection();

    connect(&ColorDialog, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(onCurrentColorChanged(const QColor&)));

    ColorDialog.setCurrentColor(m_color);
    int result = ColorDialog.exec();

    if (result == QDialog::Rejected) 
    {
        m_color = originalColor;
        emit currentColorChanged(m_color);
    }

    emit endColorSelection();

    disconnect(&ColorDialog, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(onCurrentColorChanged(const QColor&)));

    check--;
}

int QColorPushButton::getMargin() const
{
    return m_margin;
}

void QColorPushButton::setMargin(const int& margin)
{
    m_margin = margin;
    update();
}

int QColorPushButton::getRadius() const
{
    return m_radius;
}

void QColorPushButton::setRadius(const int& radius)
{
    m_radius = radius;
    update();
}

QColor QColorPushButton::getColor() const
{
    return m_color;
}

void QColorPushButton::setColor(const QColor& color, bool stopSignals)
{
    blockSignals(stopSignals);

    m_color = color;
    update();

    blockSignals(false);
}

void QColorPushButton::onCurrentColorChanged(const QColor& color)
{
    setColor(color);

    emit currentColorChanged(m_color);
}
