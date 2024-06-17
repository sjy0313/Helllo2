package ch09_nested_class.sec06_nested_interface.exam02;

public class Button {
	//정적 멤버 인터페이스 
	//ClickListener 인터페이스는 하나의 추상 메소드onClick()을 가지고 있습니다. 이 메소드는 버튼이 클릭될 때 호출될 메소드
	//
	public static interface ClickListener { 
		//추상 메소드
		void onClick();	}
	
	//필드
	private ClickListener clickListener; // callback 함수 버튼이 클릭되었을 때 호출됩니다.
	// 특정 이벤트 발생 시 동작하는 기능

		
	//메소드
	public void setClickListener(ClickListener clickListener) {
		this.clickListener = clickListener;
		//clickListener가 설정되어 있으면 onClick() 메소드를 호출합니다. 이 메소드를 통해 실제 클릭 이벤트가 처리
	}
}