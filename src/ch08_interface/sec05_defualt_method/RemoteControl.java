package ch08_interface.sec05_defualt_method;
/*
 * default method of interface : default 리턴타입 메소드(매개변수, ...) {...} 
 * - interface에는 body가 있는 메소드는 정의 불가
 * - 공통 기능을 인터페이스에 정의
 * - 기능 확장 : 기존의 코드에 영향을 주지 않으면서 호환성 유지
 * - 기능확장할 떄 default method 대신에 새로운 interface 생성을 통해 확장
 * 
  */
public interface RemoteControl {
	//상수 필드
	int MAX_VOLUME = 10;
	int MIN_VOLUME = 0;

	//추상 메소드
	void turnOn();
	void turnOff();
	void setVolume(int volume);
	// void setMute(boolean mute);

	
	//디폴트 인스턴스 메소드
	// 추상메소드를 만들어 새로 추가 한다면
	// 인터페이스를 보완할 떄 호환성의 유지 위해.
	default void setMute(boolean mute) {
		if(mute) {
			System.out.println("무음 처리합니다.");
			//추상 메소드 호출하면서 상수 필드 사용
			setVolume(MIN_VOLUME);
		} else {
			System.out.println("무음 해제합니다.");
		}
	}
}