
// Begin Definition
#ifndef QCOLOR_PUSH_BUTTON_H
#define QCOLOR_PUSH_BUTTON_H

#include <QPushButton>

// Pushbutton with color selection dialogue and swatch
class QColorPushButton : public QPushButton
{
	Q_OBJECT

public:
	QColorPushButton(QWidget* pParent = NULL);

	virtual QSize sizeHint() const;
	virtual void paintEvent(QPaintEvent* pPaintEvent);
	virtual void mousePressEvent(QMouseEvent* pEvent);

	int		getMargin() const;
	void	setMargin(const int& margin);
	int		getRadius() const;
	void	setRadius(const int& radius);
	QColor	getColor() const;
	void	setColor(const QColor& color, bool stopSignals = false);

private slots:
	void onCurrentColorChanged(const QColor& color);

signals:
	void currentColorChanged(const QColor&);

private:
	int		m_margin;
	int		m_radius;
	QColor	m_color;
};

// End definition
#endif // QCOLOR_PUSH_BUTTON_H